#!/usr/bin/env python3
"""Neural object segmentation (Layer 2 of the map's ML backbone).

Runs a tiny YOLOv8n-seg TensorRT engine on the left rectified image and
publishes a per-pixel OBJECT CLASS MASK.  semantic_filter.py fuses that mask
with its plane geometry:

    - pixels the net recognises as an object (chair, couch, table, tv, plant,
      person, ...) are TRUSTED -> their depth is kept even if thin/sparse.
    - a floating clump that is neither on a structural plane NOR inside any
      recognised object is confidently a stereo artefact -> dropped.

Why this split: walls / floor / desk are "stuff" (planes) and are handled well
by the geometric backbone; furniture is "things" and is exactly what an
instance segmenter is good at.  Together they cover the room and generalise to
any room, which is the point.

Streamlined for the 8 GB Orin Nano:
    * yolov8n-seg (~3.4 M params) at 480, FP16 TensorRT.
    * runs at SEG_HZ (default 3) and the mask is cached between frames -- the
      map only needs it as a slowly-changing gate, not every depth frame.
    * pure-TensorRT runtime: the host `tensorrt` python module is mounted in,
      CUDA memcpy goes through libcudart via ctypes -> no torch / onnxruntime /
      pycuda at runtime.

Engine is built once on the host (see models/build_engine.sh).  Topic out:
    /semantic/objects   sensor_msgs/Image  mono8, class_id+1 (0 = background)
"""
import ctypes
import os
import time

import numpy as np
import rclpy
import tensorrt as trt
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

ENGINE   = os.environ.get("SEG_ENGINE", "/workspace/models/yolov8n-seg.engine")
IMGSZ    = int(os.environ.get("SEG_IMGSZ", "480"))
SEG_HZ   = float(os.environ.get("SEG_HZ", "3"))
CONF     = float(os.environ.get("SEG_CONF", "0.25"))
IOU      = float(os.environ.get("SEG_IOU", "0.45"))
MASK_THR = float(os.environ.get("SEG_MASK_THR", "0.5"))

# COCO-80 class names (index = model class id)
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light "
        "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
        "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
        "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
        "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase "
        "scissors teddy_bear hair_drier toothbrush").split()

# ---- minimal CUDA runtime via ctypes (no pycuda/cuda-python needed) ----------
_cu = ctypes.CDLL("libcudart.so.12")
_cu.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cu.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cu.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.c_int, ctypes.c_void_p]
_cu.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cu.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cu.cudaDeviceGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                 ctypes.POINTER(ctypes.c_int)]
_cu.cudaStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                             ctypes.c_uint, ctypes.c_int]
H2D, D2H = 1, 2
_STREAM_NONBLOCKING = 1


def _make_low_priority_stream():
    """Create a CUDA stream at the LOWEST priority so cuVSLAM/VPI/nvblox GPU
    work (default priority) preempts our neural inference between kernels.
    Without this, a ~150 ms engine run stalls cuVSLAM's tracker, it drops
    frames, sees a big timestamp jump, and re-localises at a rotated pose ->
    the room duplicates at an angle. Falls back to a normal stream on error."""
    stream = ctypes.c_void_p()
    try:
        least, greatest = ctypes.c_int(), ctypes.c_int()
        _cu.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest))
        # cudaDeviceGetStreamPriorityRange returns leastPriority first, which is
        # the numerically-largest value = the LOWEST scheduling priority.
        if _cu.cudaStreamCreateWithPriority(ctypes.byref(stream),
                                            _STREAM_NONBLOCKING, least.value) == 0:
            return stream
    except Exception:
        pass
    _cu.cudaStreamCreate(ctypes.byref(stream))
    return stream


def _malloc(nbytes):
    p = ctypes.c_void_p()
    if _cu.cudaMalloc(ctypes.byref(p), nbytes) != 0:
        raise RuntimeError("cudaMalloc failed")
    return p


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes, scores, iou_thr):
    """xyxy boxes, numpy NMS -> kept indices."""
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class TRTEngine:
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = _make_low_priority_stream()
        self.io = []          # (name, is_input, shape, np_dtype, host, dev, nbytes)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_in = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(self.engine.get_tensor_shape(name))
            dt = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, np.dtype(dt))
            dev = _malloc(host.nbytes)
            self.ctx.set_tensor_address(name, int(dev.value))
            self.io.append([name, is_in, shape, dt, host, dev, host.nbytes])

    def infer(self, inp):
        outs = {}
        for e in self.io:
            name, is_in, shape, dt, host, dev, nb = e
            if is_in:
                np.copyto(host, inp.astype(dt, copy=False).reshape(shape))
                _cu.cudaMemcpyAsync(dev, host.ctypes.data, nb, H2D, self.stream)
        self.ctx.execute_async_v3(int(self.stream.value))
        for e in self.io:
            name, is_in, shape, dt, host, dev, nb = e
            if not is_in:
                _cu.cudaMemcpyAsync(host.ctypes.data, dev, nb, D2H, self.stream)
        _cu.cudaStreamSynchronize(self.stream)
        for e in self.io:
            if not e[1]:
                outs[e[0]] = e[4]
        return outs


class ObjectSeg(Node):
    def __init__(self):
        super().__init__("object_seg")
        self.bridge = CvBridge()
        self.get_logger().info(f"loading TRT engine {ENGINE} ...")
        self.trt = TRTEngine(ENGINE)
        # identify det (3-D) vs proto (4-D) outputs by rank
        self.det_key = self.proto_key = None
        for e in self.trt.io:
            if e[1]:
                continue
            if len(e[2]) == 4:
                self.proto_key = e[0]
            else:
                self.det_key = e[0]
        self.nc = len(COCO)
        self._t = 0.0
        self._tlog = 0.0

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub = self.create_publisher(Image, "/semantic/objects", qos)
        self.create_subscription(Image, "/stereo/left/image_rect", self._on_img, qos)
        self.get_logger().info(
            f"object_seg up: imgsz={IMGSZ} seg_hz={SEG_HZ} conf={CONF} "
            f"det={self.det_key} proto={self.proto_key}")

    def _letterbox(self, gray):
        """mono8 HxW -> (blob 1x3xSxS float32, r, padx, pady, out_h, out_w)."""
        h, w = gray.shape
        r = min(IMGSZ / h, IMGSZ / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        # nearest-neighbour resize (cheap, no cv2.cuda needed)
        ys = (np.arange(nh) / r).astype(int).clip(0, h - 1)
        xs = (np.arange(nw) / r).astype(int).clip(0, w - 1)
        small = gray[ys][:, xs]
        canvas = np.full((IMGSZ, IMGSZ), 114, np.uint8)
        padx, pady = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
        canvas[pady:pady + nh, padx:padx + nw] = small
        blob = (canvas.astype(np.float32) / 255.0)[None, None]      # 1x1xSxS
        blob = np.repeat(blob, 3, axis=1)                            # gray->3ch
        return blob, r, padx, pady, nh, nw

    def _on_img(self, msg):
        now = time.time()
        if now - self._t < 1.0 / SEG_HZ:
            return
        self._t = now
        try:
            self._process(msg)
        except Exception as e:
            self.get_logger().warn(f"seg skip: {e}")

    def _process(self, msg):
        gray = self.bridge.imgmsg_to_cv2(msg, "mono8")
        H, W = gray.shape
        blob, r, padx, pady, nh, nw = self._letterbox(gray)

        out = self.trt.infer(blob)
        det = out[self.det_key]      # (1, 4+nc+32, 8400)
        proto = out[self.proto_key]  # (1, 32, mh, mw)
        det = det[0].T               # (8400, 116)
        proto = proto[0]             # (32, mh, mw)
        nm = proto.shape[0]
        boxes = det[:, :4]
        cls_sc = det[:, 4:4 + self.nc]
        coeffs = det[:, 4 + self.nc:4 + self.nc + nm]

        cid = cls_sc.argmax(1)
        conf = cls_sc[np.arange(len(cid)), cid]
        m = conf > CONF
        if not m.any():
            self._publish(np.zeros((nh, nw), np.uint8), pady, nh, nw, r, H, W, [])
            return
        boxes, conf, cid, coeffs = boxes[m], conf[m], cid[m], coeffs[m]
        # cxcywh -> xyxy (letterbox coords)
        xy = np.empty_like(boxes)
        xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2; xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2; xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        keep = _nms(xy, conf, IOU)
        xy, conf, cid, coeffs = xy[keep], conf[keep], cid[keep], coeffs[keep]

        # assemble mask in letterbox space (nh x nw, unpadded region)
        mh, mw = proto.shape[1], proto.shape[2]
        pf = proto.reshape(nm, -1)               # (32, mh*mw)
        classmask = np.zeros((nh, nw), np.uint8)
        names = []
        # process biggest/most-confident first so small objects draw on top
        for k in np.argsort(conf):
            mk = _sigmoid(coeffs[k] @ pf).reshape(mh, mw)   # (mh, mw) in [0,1]
            # proto is at IMGSZ/4 over the full letterbox; sample into nh x nw
            gy = ((np.arange(nh) + pady) * mh / IMGSZ).astype(int).clip(0, mh - 1)
            gx = ((np.arange(nw) + padx) * mw / IMGSZ).astype(int).clip(0, mw - 1)
            mfull = mk[gy][:, gx] > MASK_THR
            # clip to the detection box
            bx1 = int(np.clip(xy[k, 0] - padx, 0, nw)); by1 = int(np.clip(xy[k, 1] - pady, 0, nh))
            bx2 = int(np.clip(xy[k, 2] - padx, 0, nw)); by2 = int(np.clip(xy[k, 3] - pady, 0, nh))
            box = np.zeros((nh, nw), bool); box[by1:by2, bx1:bx2] = True
            classmask[mfull & box] = cid[k] + 1
            names.append(COCO[cid[k]])
        self._publish(classmask, pady, nh, nw, r, H, W, names)

    def _publish(self, classmask, pady, nh, nw, r, H, W, names):
        # classmask is nh x nw (unpadded letterbox). That already equals the
        # camera FOV at reduced res -> publish as-is; semantic_filter nearest-
        # resamples it to its own processing grid.
        m = self.bridge.cv2_to_imgmsg(classmask, "mono8")
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "camera_left_optical"
        self.pub.publish(m)
        now = time.time()
        if now - self._tlog > 5.0:
            uniq = sorted(set(names))
            self.get_logger().info(
                f"objects: {len(names)} dets {('['+', '.join(uniq)+']') if uniq else '(none)'}")
            self._tlog = now


def main():
    rclpy.init()
    node = ObjectSeg()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
