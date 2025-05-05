# clean this up a bit maybe and add a basic gui or smth

import time

import cv2
import mss
import numpy

# for safe shutdown
from pynput import keyboard
import time

# for saving image
from PIL import Image

# import pyautogui
# pos = pyautogui.mouseinfo.position()


resolution=(2560,1440)
# x = numpy.arange(0, 300)
# y = numpy.arange(0, 300)
# X, Y = numpy.meshgrid(x, y)
# Z = X + Y
# Z = 255*Z/Z.max()
# surf = pygame.surfarray.make_surface(Z)


N = 30
last_n = [None for _ in range(N)]
idx = 0
idx_up=False
break_program = False
is_saving=False
# just how much ass are we talking about?


def on_press(key):
    global break_program, idx, is_saving
    # print (key)
    if key == keyboard.Key.esc:
        # print ('end pressed')
        break_program = True
    elif key == keyboard.Key.print_screen:
        curr_idx=idx
        print("Started savin")
        is_saving=True
        for i in range(len(last_n)):
            idx_modn = (curr_idx + i) + int(idx_up)
            im = last_n[idx_modn % N]
            # im.save(f"img_{i}_{curr_idx=}.jpeg")
            if im is not None:
                im=Image.frombytes(mode="RGB", size=resolution, data=im.rgb)
                im.save(f"test/img_{i}.jpeg", format='JPEG', subsampling=0, quality=95)
        print("Finished savin")
        is_saving=False

    return not break_program


# important note -
# rat location issue in windows - this should work in linux (using the with-cursor arg)
# currently cursor does NOT appear
prev_avg=0
totes=0
epsilon=1e-5
show_fps_interval=5
dont_show_fps=0
# added keyboard control so that the thing will stop running when you click the ting and ting
with keyboard.Listener(on_press=on_press) as listener, mss.mss() as sct:
    # with mss.mss() as sct:
    # this method for filling img array seems to be faster, more testing required.
    img = numpy.empty((resolution[1], resolution[0], 4), dtype='uint8')
    # Part of the screen to capture
    monitor = {'top': 0, 'left': 0, 'width': resolution[0], 'height': resolution[1]}
    # while 'Screen capturing':
    while not break_program:
        last_time = time.time()
        idx_up=False
        # method 1 (sup)
        # Get raw pixels from the screen, save it to a Numpy array

        # img[:] = sct.grab(monitor)
        # instead of grabbing image, store it temporarily alongside the past 30 images
        if not is_saving:
            last_n[idx] = sct.grab(monitor)
            # increment index
            idx = (idx + 1) % N
        idx_up=True

        # method 2
        # img= numpy.array(sct.grab(monitor))
        # print(img2.shape, img.shape)
        # img.(img2)

        # the following alternative uses the .rgb property to avoid the additional conversion for the pygame backend
        # apparently though, it is much slower (27fps vs an average of ~45 in 1080p)
        # img = numpy.array(bytearray(sct.grab(monitor).rgb)).reshape((resolution[1], resolution[0], -1))
        # importantly, if you want to usethe above line for testing, make sure to remove
        # the bgra2rgb conversion later in the code

        # Display the picture
        # cv2.imshow('OpenCV/Numpy normal', img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
        if (not (dont_show_fps % show_fps_interval)) and not is_saving:
            curr = (1 / (time.time() - last_time + epsilon))  # epsilon for edge case
            totes += 1
            prev_avg = prev_avg + (curr - prev_avg) / totes
            print(f'fps: {curr}, average: {prev_avg}')
        dont_show_fps = (dont_show_fps + 1) % show_fps_interval

        # print(f"The actual frame: {img}") # (yeah right)

        # Press "q" to quit
        # fucking modified as it does not fucking work
        # it will quit on any mouse movement
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        # print(img.shape)
        # rgbImage = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # break and shiet
    listener.join()

