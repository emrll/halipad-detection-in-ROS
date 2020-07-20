#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import dronekit
import pymavlink
import gi
import numpy as np
import time
import roslib
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import math
import csv


detector=cv2.CascadeClassifier('helipad_trained_data.xml')
print(detector)
#camera
horizontal_fov = 118.2 * math.pi/180
vertical_fov = 69.5 * math.pi/180
horizontal_resolution = 1280
vertical_resolution = 720
#global image
bridge = CvBridge()

def helipad_mid_point(corners,x,y):

    x1 = x + corners[0][0][0]
    y1 = y + corners[0][0][1]
    x2 = x + corners[1][0][0]
    y2 = y + corners[1][0][1]
    x3 = x + corners[2][0][0]
    y3 = y + corners[2][0][1]
    x4 = x + corners[3][0][0]
    y4 = y + corners[3][0][1]
    x5 = x + corners[4][0][0]
    y5 = y + corners[4][0][1]
    x6 = x + corners[5][0][0]
    y6 = y + corners[5][0][1]
    x7 = x + corners[6][0][0]
    y7 = y + corners[6][0][1]
    x8 = x + corners[7][0][0]
    y8 = y + corners[7][0][1]
    x9 = x + corners[8][0][0]
    y9 = y + corners[8][0][1]
    x10 = x + corners[9][0][0]
    y10 = y + corners[9][0][1]
    x11 = x + corners[10][0][0]
    y11 = y + corners[10][0][1]
    x12 = x + corners[11][0][0]
    y12 = y + corners[11][0][1]

    xs = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]
    ys = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12]

    x_helipad_mid = int((sum(xs)) / 12)
    y_helipad_mid = int((sum(ys)) / 12)

    point1_length = math.sqrt((x1 - x_helipad_mid) ** 2 + (y1 - y_helipad_mid) ** 2)
    point2_length = math.sqrt((x2 - x_helipad_mid) ** 2 + (y2 - y_helipad_mid) ** 2)
    point3_length = math.sqrt((x3 - x_helipad_mid) ** 2 + (y3 - y_helipad_mid) ** 2)
    point4_length = math.sqrt((x4 - x_helipad_mid) ** 2 + (y4 - y_helipad_mid) ** 2)
    point5_length = math.sqrt((x5 - x_helipad_mid) ** 2 + (y5 - y_helipad_mid) ** 2)
    point6_length = math.sqrt((x6 - x_helipad_mid) ** 2 + (y6 - y_helipad_mid) ** 2)
    point7_length = math.sqrt((x7 - x_helipad_mid) ** 2 + (y7 - y_helipad_mid) ** 2)
    point8_length = math.sqrt((x8 - x_helipad_mid) ** 2 + (y8 - y_helipad_mid) ** 2)
    point9_length = math.sqrt((x9 - x_helipad_mid) ** 2 + (y9 - y_helipad_mid) ** 2)
    point10_length = math.sqrt((x10 - x_helipad_mid) ** 2 + (y10 - y_helipad_mid) ** 2)
    point11_length = math.sqrt((x11 - x_helipad_mid) ** 2 + (y11 - y_helipad_mid) ** 2)
    point12_length = math.sqrt((x12 - x_helipad_mid) ** 2 + (y12 - y_helipad_mid) ** 2)

    points = (point1_length, point2_length, point3_length, point4_length, point5_length, point6_length, point7_length, point8_length,point9_length, point10_length, point11_length, point12_length)

    max_length = max(points)

    idx_max = points.index(max_length)

    x_target_point = int(xs[idx_max])
    y_target_point = int(ys[idx_max])

    return x_helipad_mid,y_helipad_mid,x_target_point,y_target_point

def csv_write(filename,row):

    with open(filename,'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

def image_corner_detection(image,ymin,ymax,xmin,xmax):

    kesilmis_image = image[ymin:ymax, xmin:xmax]

    corners = cv2.goodFeaturesToTrack(kesilmis_image,maxCorners=50,qualityLevel=0.3,minDistance=1,useHarrisDetector=True)
        
    #kose var mı yok mu onu check eder
                
    return corners

class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self._frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK


if __name__ == '__main__':
    # Create the video object
    # Add port= if is necessary to use a different one
    video = Video()
    pub = rospy.Publisher('camera/image_raw1',Image,queue_size=10)
    pub1=rospy.Publisher('camera/image_raw',Image,queue_size=10)
    rospy.init_node('camera',anonymous=True)
    rate=rospy.Rate(15)
    print("kontrol1")
    while(1):
        # Wait for the next frame
        if not video.frame_available():
            continue

        image = video.frame()
        grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rate.sleep()
        #rospy.spin()
        start = time.time()
        #print("döngü1")
        #cv2.imshow("image",image)
        #grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        kose_sayisi = 0

      

        helipad = detector.detectMultiScale(grey_image, scaleFactor=1.1, flags=cv2.CASCADE_SCALE_IMAGE, minNeighbors=5)
        for (x, y, w, h) in helipad:

            obje_sayisi = np.size(helipad, 0)  # kac helipad var onu bulur

            ymin = int(y)
            ymax = int(y) + int(h)
            xmin = int(x)
            xmax = int(x) + int(w)

            corners = image_corner_detection(grey_image, ymin, ymax, xmin, xmax)

            kose_cifti = np.size(corners)  # kose degerlerını olusturan x ve y datalarını icerir

            if not kose_cifti == 1:

                isnan_check = np.isnan(corners)
            #                print(isnan_check,' ',type(isnan_check))

                kose_sayisi = kose_cifti / 2

                if kose_sayisi == 12:

                # goruntu uzerine kare ciz
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)

                    if not (kose_cifti == 1 and isnan_check == True):

                        for i in corners:
                            x_corner, y_corner = i.ravel()
                            x_corner = np.int0(x_corner)
                            y_corner = np.int0(y_corner)
                            cv2.circle(image, (x + x_corner, y + y_corner), 3, 255, -1)

                    x_helipad_mid,y_helipad_mid,x_target_point,y_target_point= helipad_mid_point(corners, x, y)

                    # Helipad orta noktasını göster
                    cv2.circle(image, (x_helipad_mid, y_helipad_mid), 3, 255, -1)

                    # Helipad orta nokta ile hedef nokta arasına çizgi çiz
                    cv2.line(image, (x_target_point, y_target_point), (x_helipad_mid, y_helipad_mid), (255, 0, 0), 2)

                    # Hedef çizginin x ve y deki iz düşümü
                    target_line_x_projection = abs(x_helipad_mid - x_target_point)
                    target_line_y_projection = abs(y_helipad_mid - y_target_point)

                    target_line_length_pixel = np.sqrt(np.square(target_line_x_projection) + np.square(target_line_y_projection))

                    theta = np.arctan2(target_line_y_projection, target_line_x_projection)

                    x_pixel_line = np.cos(theta) * target_line_length_pixel
                    y_pixel_line = np.sin(theta) * target_line_length_pixel

                    target_line_length_real = 167.886 / 2  # mm

                    line_x_length_real = np.cos(theta) * target_line_length_real
                    line_y_length_real = np.sin(theta) * target_line_length_real

                    # x ve y ekseninde 1 pixel in kaç mm geldiğini hesapla
                    pixel_to_real_ratio_x = line_x_length_real / x_pixel_line
                    pixel_to_real_ratio_y = line_y_length_real / y_pixel_line

                    # yüksekliği hesapla
                    h_x = (pixel_to_real_ratio_x * 1280 / (2 * 1.67087824451)) * 0.1  # cmh
                    h_y = (pixel_to_real_ratio_y * 720 / (2 * 0.6937246842)) * 0.1  # cm

                    yukseklik = h_x + h_x / 2
                    print('yükseklik: ', h_x, ' ', yukseklik)
		    yukseklik_text = 'yükesklik'+ str(yukseklik)
                    y_text='								'+yukseklik_text
                    cv2.putText(image, y_text, (30,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1,cv2.CV_AA)
                    row = [round(target_line_x_projection, 2), round(target_line_y_projection, 2), round(x_helipad_mid, 2),
                            round(y_helipad_mid, 2), target_line_length_pixel, theta * (360 / (2 * np.pi)), x_pixel_line,
                            y_pixel_line, line_x_length_real, line_y_length_real, pixel_to_real_ratio_x,
                            pixel_to_real_ratio_y, h_x, h_y]
                    csv_file_name = 'solo_veriler.csv'
                    csv_write(csv_file_name, row)

        stop = time.time()
    
        gecen_zaman = str(round((1/(stop-start)),1))  #virgülden sonra 1 hane olacak şekilde yuvarla
        gecen_zaman = 'Frame Rate: '+gecen_zaman   #2 string i birleştir
    
        kose_sayisi_text = 'Kose Sayisi: '+ str(kose_sayisi)
        
        image_text = gecen_zaman+'                  '+ kose_sayisi_text
    
        cv2.putText(image, image_text, (30,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1,cv2.CV_AA)
        msg_frame = CvBridge().cv2_to_imgmsg(image,"bgr8")
        pub.publish(msg_frame)
        
        msg_frame1 = CvBridge().cv2_to_imgmsg(grey_image,"mono8")
        pub1.publish(msg_frame1)
        
        #cv2.imshow('orjinal',image)


        #groundspeed = round(vehicle.groundspeed,3)
        #heading =  round(vehicle.heading,3)
        #z_velocity =  vehicle.mode.name
        #aci = vehicle.attitude
        #relatif_yukseklik = round(vehicle.location.global_relative_frame.alt,3)
        #global_yukseklik = round(vehicle.location.global_frame.alt,3)
    
        #print("relatif_yukseklik: ", relatif_yukseklik, " global_yukseklik: ",global_yukseklik )

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

