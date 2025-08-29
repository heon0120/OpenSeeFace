import re
import sys
import os
import cv2
import numpy as np
import time
import traceback
import gc

# Libraries exclusive to the Windows environment may cause errors when imported, so they are wrapped in try-except.
try:
    import escapi
except ImportError:
    escapi = None
try:
    import dshowcapture
except ImportError:
    dshowcapture = None
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class VideoReader():
    def __init__(self, capture, camera=False):
        if os.name == 'nt' and camera:
            self.cap = cv2.VideoCapture(capture, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(capture)
        if self.cap is None:
            print("The video source cannot be opened")
            sys.exit(0)
        self.name = str(capture)

    def is_open(self):
        return self.cap.isOpened()

    def is_ready(self):
        return True

    def read(self):
        return self.cap.read()

    def close(self):
        self.cap.release()


class EscapiReader(VideoReader):
    def __init__(self, capture, width, height, fps):
        if escapi is None:
            raise ImportError("escapi module is not available.")
        self.device = None
        self.width = width
        self.height = height
        self.fps = fps
        self.device = capture
        escapi.count_capture_devices()
        self.name = str(escapi.device_name(self.device).decode('utf8', 'surrogateescape'))
        self.buffer = escapi.init_camera(self.device, self.width, self.height, self.fps)
        escapi.do_capture(self.device)

    def is_open(self):
        return True

    def is_ready(self):
        return escapi.is_capture_done(self.device)

    def read(self):
        if escapi.is_capture_done(self.device):
            image = escapi.read(self.device, self.width, self.height, self.buffer)
            escapi.do_capture(self.device)
            return True, image
        else:
            return False, None

    def close(self):
        if escapi:
            escapi.deinit_camera(self.device)


class DShowCaptureReader(VideoReader):
    def __init__(self, capture, width, height, fps, use_dshowcapture=True, dcap=None):
        if dshowcapture is None:
            raise ImportError("dshowcapture module is not available.")
        self.device = None
        self.width = width
        self.height = height
        self.fps = fps
        self.dcap = dcap;
        self.device = dshowcapture.DShowCapture()
        self.device.get_devices()
        info = self.device.get_info()
        self.name = info[capture]['name']
        if info[capture]['type'] == "Blackmagic":
            self.name = "Blackmagic: " + self.name
            if dcap is None or dcap < 0:
                dcap = 0
        ret = False
        if dcap is None:
            ret = self.device.capture_device(capture, self.width, self.height, self.fps)
        else:
            if dcap < 0:
                ret = self.device.capture_device_default(capture)
            else:
                ret = self.device.capture_device_by_dcap(capture, dcap, self.width, self.height, self.fps)
        if not ret:
            raise Exception("Failed to start capture.")
        self.width = self.device.width
        self.height = self.device.height
        self.fps = self.device.fps
        print(
            f"Camera: \"{self.name}\" Capability ID: {dcap} Resolution: {self.device.width}x{self.device.height} Frame rate: {self.device.fps} Colorspace: {self.device.colorspace} Internal: {self.device.colorspace_internal} Flipped: {self.device.flipped}")
        self.timeout = 1000

    def is_open(self):
        return self.device.capturing()

    def is_ready(self):
        return self.device.capturing()

    def read(self):
        img = None
        try:
            img = self.device.get_frame(self.timeout)
        except:
            gc.collect()
            img = self.device.get_frame(self.timeout)
        if img is None:
            return False, None
        else:
            return True, img

    def close(self):
        if self.device:
            self.device.destroy_capture()


class OpenCVReader(VideoReader):
    def __init__(self, capture, width, height, fps):
        self.device = None
        self.width = width
        self.height = height
        self.fps = fps
        self.name = str(capture)
        super(OpenCVReader, self).__init__(capture, camera=True)
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.cap.set(38, 1)

    def is_open(self):
        return super(OpenCVReader, self).is_open()

    def is_ready(self):
        return super(OpenCVReader, self).is_ready()

    def read(self):
        return super(OpenCVReader, self).read()

    def close(self):
        super(OpenCVReader, self).close()


class RawReader:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)

        if self.width < 1 or self.height < 1:
            print("No acceptable size was given for reading raw RGB frames.")
            sys.exit(0)

        self.len = self.width * self.height * 3
        self.open = True

    def is_open(self):
        return self.open

    def is_ready(self):
        return True

    def read(self):
        frame = bytearray()
        read_bytes = 0
        while read_bytes < self.len:
            bytes = sys.stdin.buffer.read(self.len)
            read_bytes += len(bytes)
            frame.extend(bytes)
        return True, np.frombuffer(frame, dtype=np.uint8).reshape((self.height, self.width, 3))

    def close(self):
        self.open = False


class RealSenseReader:
    def __init__(self, width, height, fps):
        if rs is None:
            raise ImportError(
                "pyrealsense2 module is not available. Please install it with 'pip install pyrealsense2'.")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.name = "RealSense Camera"
        self.width = width
        self.height = height
        self.fps = fps
        self.is_pipeline_started = False

        # Set Color stream
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # Set Depth stream
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        try:
            # realsense pipeline start
            self.profile = self.pipeline.start(self.config)
            self.is_pipeline_started = True

            # Wait a moment for the camera to stabilize
            time.sleep(2)

            # Discard a few frames to warm up the camera
            for _ in range(10):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    if frames:
                        break
                except:
                    continue

            print(f"RealSense camera started successfully. Resolution: {width}x{height}, FPS: {fps}")
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")
            self.is_pipeline_started = False
            raise RuntimeError("RealSense camera failed to start.")

    def is_open(self):
        return self.is_pipeline_started and self.pipeline is not None

    def is_ready(self):
        return self.is_pipeline_started

    def read(self):
        if not self.is_pipeline_started:
            return False, None

        try:
            # Set a timeout to wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # 5초 타임아웃

            if not frames:
                print("No frames received from RealSense camera")
                return False, None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                print("No color frame available")
                return False, None

            # Convert frame data to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Check Frame Size
            if frame is None or frame.size == 0:
                print("Empty frame received")
                return False, None

            # Check Frame Type
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Unexpected frame shape: {frame.shape}")
                return False, None

            # Check BGR
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            return True, frame

        except rs.error as e:
            print(f"RealSense error: {e}")
            return False, None
        except Exception as e:
            print(f"General error reading RealSense frame: {e}")
            traceback.print_exc()
            return False, None

    def close(self):
        if self.is_pipeline_started and self.pipeline:
            try:
                self.pipeline.stop()
                self.is_pipeline_started = False
                print("RealSense camera stopped.")
            except Exception as e:
                print(f"Error stopping RealSense pipeline: {e}")


def try_int(s):
    try:
        return int(s)
    except:
        return None


def test_reader(reader):
    got_any = 0
    try:
        for i in range(30):
            if not reader.is_ready():
                time.sleep(0.1)  # add waiting time for realsense
                continue

            ret, frame = reader.read()
            if not ret:
                time.sleep(0.1)
                print(f"No frame at attempt {i}")
            else:
                print(f"Got frame {i + 1}, shape: {frame.shape if frame is not None else 'None'}")
                got_any += 1
                if got_any > 5:  # Fewer frames are enough for testing
                    break

        if reader.is_open():
            success = got_any > 0
            print(f"Test reader result: {success}, got {got_any} frames")
            return success
        print("Reader not open - Fail")
        return False
    except Exception as e:
        print(f"Test reader exception: {e}")
        traceback.print_exc()
        return False


class InputReader():
    def __init__(self, capture, raw_rgb, width, height, fps, use_dshowcapture=False, dcap=None, use_realsense=False):
        self.reader = None
        self.name = str(capture)

        try:
            if raw_rgb > 0:
                print("Using raw RGB input...")
                self.reader = RawReader(width, height)
            elif use_realsense:
                print("Attempting to use RealSense camera...")
                self.reader = RealSenseReader(width, height, fps)
                self.name = self.reader.name

                # RealSense reader test
                print("Testing RealSense reader...")
                if test_reader(self.reader):
                    print("RealSense reader test passed")
                    return
                else:
                    print("RealSense reader test failed")
                    raise RuntimeError("RealSense camera test failed")

            elif os.path.exists(capture):
                print(f"Using video file: {capture}")
                self.reader = VideoReader(capture)
            elif capture == str(try_int(capture)):
                if os.name == 'nt':
                    # try DShowCapture
                    good = True
                    name = ""
                    try:
                        if use_dshowcapture and dshowcapture:
                            self.reader = DShowCaptureReader(int(capture), width, height, fps, dcap=dcap)
                            name = self.reader.name
                            good = test_reader(self.reader)
                            self.name = name
                        else:
                            good = False
                    except:
                        print("DShowCapture exception: ")
                        traceback.print_exc()
                        good = False
                    if good:
                        return
                    # try Escapi
                    good = True
                    try:
                        if escapi:
                            print(f"DShowCapture failed. Falling back to escapi for device {name}.", file=sys.stderr)
                            escapi.init()
                            devices = escapi.count_capture_devices()
                            found = None
                            for i in range(devices):
                                escapi_name = str(escapi.device_name(i).decode('utf8', 'surrogateescape'))
                                if name == escapi_name:
                                    found = i
                            if found is None:
                                good = False
                            else:
                                print(f"Found device {name} as {i}.", file=sys.stderr)
                                self.reader = EscapiReader(found, width, height, fps)
                                good = test_reader(self.reader)
                        else:
                            good = False
                    except:
                        print("Escapi exception: ")
                        traceback.print_exc()
                        good = False
                    if good:
                        return
                    # try opencv
                    print(f"Escapi failed. Falling back to OpenCV. If this fails, please change your camera settings.",
                          file=sys.stderr)
                    self.reader = OpenCVReader(int(capture), width, height, fps)
                    self.name = self.reader.name
                else:
                    self.reader = OpenCVReader(int(capture), width, height, fps)
        except Exception as e:
            print("Error: " + str(e))
            traceback.print_exc()

        if self.reader is None or not self.reader.is_open():
            print("There was no valid input.")
            sys.exit(0)

    def is_open(self):
        return self.reader.is_open()

    def is_ready(self):
        return self.reader.is_ready()

    def read(self):
        return self.reader.read()

    def close(self):
        self.reader.close()