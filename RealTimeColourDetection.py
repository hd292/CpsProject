import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import time
import math

#Definition of  motor pin 
IN1 = 20
IN2 = 21
IN3 = 19
IN4 = 26
ENA = 16
ENB = 13



EchoPin = 0
TrigPin = 1
GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)

#initialisation of pins
def motor_init():
    global pwm_ENA
    global pwm_ENB
    global delaytime
    GPIO.setup(ENA,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(ENB,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(EchoPin,GPIO.IN)
    GPIO.setup(TrigPin,GPIO.OUT)
    #Set the PWM pin and frequency is 2000hz
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    pwm_ENA.start(0)
    pwm_ENB.start(0)

def run(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)


def left(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)

def right(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(20)
    pwm_ENB.ChangeDutyCycle(20)
    time.sleep(delaytime)

def spin_left(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(10)
    pwm_ENB.ChangeDutyCycle(10)
    time.sleep(delaytime)

def spin_right(delaytime):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(5)
    pwm_ENB.ChangeDutyCycle(5)
    time.sleep(delaytime)

def brake(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(5)
    pwm_ENB.ChangeDutyCycle(5)
    time.sleep(delaytime)

time.sleep(2)
CAMERA_DEVICE_ID = 0
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
fps = 0

hsv_min = np.array((50, 80, 80))
hsv_max = np.array((120, 255, 255))

colors = []

#https://github.com/automaticdai/rpi-object-detection/blob/7ad100d085134414caae7ca8931dfa7a023a09eb/src/object-tracking-color/cv_object_tracking_color.py
#We used this code to get the colour value for our camera we have altered accordingly to fit our needs 
def isset(v):
    try:
        type (eval(v))
    except:
        return 0
    else:
        return 1


def on_mouse_click(event, x, y, flags, frame):
    global colors

    if event == cv2.EVENT_LBUTTONUP:
        color_bgr = frame[y, x]
        color_rgb = tuple(reversed(color_bgr))
        #frame[y,x].tolist()

        print(color_rgb)

        color_hsv = rgb2hsv(color_rgb[0], color_rgb[1], color_rgb[2])
        print(color_hsv)

        colors.append(color_hsv)

        print(colors)


# R, G, B values are [0, 255]. 
# Normally H value is [0, 359]. S, V values are [0, 1].
# However in opencv, H is [0,179], S, V values are [0, 255].
# Reference: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
def hsv2rgb(h, s, v):
    # Ensure h is in the correct range of 0-360 degrees
    h = float(h)
    
    # Normalize s and v from 0-255 to 0-1
    s = float(s) / 255
    v = float(v) / 255
    
    # Handle case where s is 0 (this means the color is a shade of gray)
    if s == 0:
        r = g = b = int(v * 255)
        return (r, g, b)
    
    # Convert hue to the range [0, 6)
    h60 = h / 60.0
    hi = int(h60) % 6
    f = h60 - math.floor(h60)
    
    # Calculate intermediary values for the RGB calculation
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    # Compute the RGB values based on the section of the color wheel
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    
    # Convert from the range 0-1 back to 0-255 and round to nearest integer
    r, g, b = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    
    return (r, g, b)


def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx

    h = int(h / 2)
    s = int(s * 255)
    v = int(v * 255)

    return (h, s, v)


def visualize_fps(image, fps: int):
    if len(np.shape(image)) < 3:
        text_color = (255, 255, 255)  # white
    else:
        text_color = (0, 255, 0)  # green
    row_size = 20  # pixels
    left_margin = 24  # pixels

    font_size = 1
    font_thickness = 1

    # Draw the FPS counter
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    return image
    
#using the thresh from automaticdai github we calculate the amount fo pixels that white this is used on 3 different sections which are defined later and input in this function
def calculate_white_percentage(binary_image):
    # Ensure binary image is binary
    if len(binary_image.shape) > 2:  # Check if image has multiple channels
        raise ValueError("Input image must be binary (single channel)")

    total_pixels = binary_image.size
    white_pixels = np.sum(binary_image == 255)

    percentage_white = (white_pixels / total_pixels) * 100
    return percentage_white
sectionWidth = IMAGE_WIDTH // 3



def Distance():
    GPIO.output(TrigPin,GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin,GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin,GPIO.LOW)

    t3 = time.time()

    while not GPIO.input(EchoPin):
        t4 = time.time()
        if (t4 - t3) > 0.03 :
            return -1


    t1 = time.time()
    while GPIO.input(EchoPin):
        t5 = time.time()
        if(t5 - t1) > 0.03 :
            return -1

    t2 = time.time()
    time.sleep(0.01)
    return ((t2 - t1)* 340 / 2) * 100
    
    
def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
            distance = Distance()
            while int(distance) == -1 :
                distance = Distance()
                print("Tdistance is %f"%(distance) )
            while (int(distance) >= 500 or int(distance) == 0) :
                distance = Distance()
                print("Edistance is %f"%(distance) )
            ultrasonic.append(distance)
            num = num + 1
            time.sleep(0.01)
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3])/3
    print("distance is %f"%(distance) ) 
    return distance



start = False

if __name__ == "__main__":
    try:
        # create video capture
        cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
        motor_init()

        # set resolution to 320x240 to reduce latency 
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)

        while True:
            # ----------------------------------------------------------------------
            # record start time
            start_time = time.time()
            # Read the frames frome a camera
            _, frame = cap.read()
            frame = cv2.blur(frame,(3,3))

            # Or get it from a JPEG
            # frame = cv2.imread('frame0010.jpg', 1)

            # Convert the image to hsv space and find range of colors
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.namedWindow('frame')
            cv2.setMouseCallback('frame', on_mouse_click, frame)

            # find the color using a color threshhold
            if colors:
                # find max & min h, s, v
                minh = min(c[0] for c in colors)
                mins = min(c[1] for c in colors)
                minv = min(c[2] for c in colors)
                maxh = max(c[0] for c in colors)
                maxs = max(c[1] for c in colors)
                maxv = max(c[2] for c in colors)

                print("New HSV threshold: ", (minh, mins, minv), (maxh, maxs, maxv))
                hsv_min = np.array((minh, mins, minv))
                hsv_max = np.array((maxh, maxs, maxv))  

            thresh = cv2.inRange(hsv, hsv_min, hsv_max)
            thresh2 = thresh.copy()
            thresh3 = thresh.copy()


            # find contours in the threshold image
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            #print(major_ver, minor_ver, subminor_ver)
            
            

            # findContours() has different form for opencv2 and opencv3
            if major_ver == "2" or major_ver == "3":
                _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # finding contour with maximum area and store it as best_cnt
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    best_cnt = cnt

            # finding centroids of best_cnt and draw a circle there
            if isset('best_cnt'):
                M = cv2.moments(best_cnt)
                cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                cv2.circle(frame,(cx,cy),5,255,-1)
                print("Central pos: (%d, %d)" % (cx,cy))
            else:
                print("[Warning]Tag lost...")
            leftSection = thresh[:, :sectionWidth]
            middleSection = thresh[:, sectionWidth:2*sectionWidth]
            rightSection = thresh[:, 2*sectionWidth:]
            # Show the original and processed image
            #res = cv2.bitwise_and(frame, frame, mask=thresh2)
            #determine white percentage and move the robot accordingly 
            white_percentage_left= calculate_white_percentage(leftSection)
            white_percentage_middle= calculate_white_percentage(middleSection)
            white_percentage_right= calculate_white_percentage(rightSection)

            print(f"Percentage of left white pixels: {white_percentage_left:.2f}%")
            print(f"Percentage of middle white pixels: {white_percentage_middle:.2f}%")
            print(f"Percentage of Right white pixels: {white_percentage_right:.2f}%")

            cv2.imshow('frame', visualize_fps(frame, fps))
            cv2.imshow('thresh', visualize_fps(thresh2, fps))
            
            print(time)
            if cv2.waitKey(30) == 13 or start == True:
                start = True
                distanceFromCar  = Distance_test()
                if distanceFromCar < 10:
                    brake(0.1)
                    print("collision avoided")
                    if white_percentage_left>white_percentage_middle and white_percentage_left > white_percentage_right:
                        print("left")
                        spin_left(0.01)
                    elif white_percentage_right > white_percentage_left and white_percentage_right > white_percentage_middle:
                        print("right")
                        spin_right(0.01)
                elif white_percentage_left>70 and white_percentage_middle > 70 and white_percentage_right:
                    print("stop")
                    brake(0.1)
                elif white_percentage_left>white_percentage_middle and white_percentage_left > white_percentage_right:
                    print("left")
                    spin_left(0.01)
                elif white_percentage_middle > white_percentage_left and white_percentage_middle > white_percentage_right:
                    print("run")                
                    run(0.1)
                elif white_percentage_right > white_percentage_left and white_percentage_right > white_percentage_middle:
                    print("right")
                    spin_right(0.01)
            # ----------------------------------------------------------------------
            # record end time
            end_time = time.time()
            # calculate FPS and cap FPS
            seconds = end_time - start_time
            fps = 1.0 / seconds
            #we cap the fps to allow the motors to run for a bit in addition to not overwhelming the motor function as when fps is too high the car will over correct
            if fps < 0.09:
                fps = 0.09
            
            print("Estimated fps:{0:0.1f}".format(fps));
            # if key pressed is 'Esc' then exit the loop
            if cv2.waitKey(33) == 27:
                break
    except Exception as e:
        print(e)
    finally:
        # Clean up and exit the program
        cv2.destroyAllWindows()
        cap.release()
        pwm_ENA.stop()
        pwm_ENB.stop()
        GPIO.cleanup()
