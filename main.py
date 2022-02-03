from pytube import YouTube
import sys
import cv2
import numpy as np
from itertools import islice
from csv import reader
import pytesseract as pt
from numpy import array


class ChaseAnalyser():
    GREEN = 'green'
    LBLUE = 'lightblue'
    DBLUE = 'darkblue'
    GREEN = 'moneyboxgreen'
    RED = 'red'
    BLACK = 'black'
    QBLUE = 'question blue'
    ChaserRight = 'Chaser gets answer right'
    ChaserWrong = 'Chaser gets answer wrong'
    ContestantRight = 'Contestant gets answer right'
    ContestantWrong = 'Contestant gets answer wrong'
    Neither = 'Neither get answer right'
    Both = 'Both get answer right'
    ARight = 'A correct'
    BRight = 'B correct'
    CRight = 'C correct'
    Placeholder = 'Placeholder'
    TopChoice = 'Top choice'
    MidChoice = 'Middle choice'
    LowChoice = 'Low choice'
    frames_to_skip = 2

    # download videos (high res)
    def download_videos(self):
        with open(sys.argv[1]) as f:
            lines = f.readlines()
        for url in lines:
            yt = YouTube(url)
            streams = yt.streams.filter(progressive=True)
            itag = streams[-1].itag
            stream = yt.streams.get_by_itag(itag)
            stream.download()

    # get frames from the videos
    def get_frames(self, stream):
        video = cv2.VideoCapture(stream)
        i = -1
        repeater = 0
        last_frame = None
        all_money = None
        green_box = None
        # Loop through each frame in the video.
        while video.isOpened():
            i += 1
            ret, frame_bgr = video.read()
            if not ret:
                break
            else:
                # Convert the frame into RGB colour format.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Skip some frames for efficiency.
                if i % self.frames_to_skip == 0:

                # get options for the money
                if x.strip_option_blue(375, 450, 830, frame_rgb) and x.strip_blue(440, 450, 830, frame_rgb) and x.strip_option_blue(490, 450, 830, frame_rgb, 70):
                    if all_money is not None:

                        cv2.imwrite(F"all_options{i}.png", all_money)
                        y = NumberAnalyser()
                        all_money = cv2.cvtColor(all_money, cv2.COLOR_BGR2RGB)
                        options = y.numbers(all_money)
                        print(i, options)
                        all_money = None
                        continue
                elif x.strip_mboxgreen(375, 450, 830, frame_rgb, 70) or x.strip_mboxgreen(440, 450, 830,frame_rgb,70) or x.strip_mboxgreen(490, 450, 830, frame_rgb, 70):
                    all_money = frame_bgr
                    if green_box is not None:
                         # cv2.imwrite(F"selected_option{i}.png", green_box)
                        continue

                    # get question boxes (with no options) to use as signal to get previous question's last frame
                    if x.strip_blue(590, 230, 1050, frame_rgb):
                        # Skip if there is no previously saved frame to analyse.
                        if last_frame is None:
                            continue

                        # We have the final frame from the last question, where all of the choices should be present.

                        # Write the frame to file for debugging/later use (still in BGR as imwrite requires).
                        cv2.imwrite(F"pipeline/{i}.png", last_frame)

                        # Convert the frame to RGB to analyse.
                        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

                        # Finds the sequence of colours corresponding to the given answers from Chaser and Contestant.
                        answer_sequence = x.logic(last_frame)
                        # Takes the colour sequence and infers whether the Chaser & Contestant were right or wrong.
                        contestant_correct, chaser_correct = x.inference(answer_sequence)
                        print('frame ' + str(i))
                        print(f'Chaser was correct is {chaser_correct}\nContestant is correct is {contestant_correct}')

                        # Reset to find the next question.
                        last_frame = None
                        continue # need to update i or counter or smth

                    # Checks for a question box which has the three options also present.
                    if x.strip_blue(534, 230, 1050, frame_rgb):
                        # Checks that the answer has been given -> green box present.
                        if x.strip_green(640, 230, 1050, frame_rgb, 0.3):
                            print('answer found')
                            last_frame = frame_bgr
                            # We have found a frame with a question with the green answer box.
                            # Save the current frame for debugging/later use.
                            cv2.imwrite(F"green/{i}.png", frame_bgr)

                            repeater = i
                            # elif x.area(640, 230, 1050, 25, frames, ChaseAnalyser.GREEN):
                            #     last_frame = frame

                    # get question boxes (with options)
                    # if x.strip_blue(534, 230, 1050, frames, 0.3):
                    #     if i - repeater < 50:
                    #         continue
                        # get question boxes with a GREEN option
                        # We have verified that a question is on screen, now want to check if the green answer is present
        video.release()
        cv2.destroyAllWindows()

    # Checks if a frame contains a question box with the three options. Returns True if so. False otherwise.
    # Question box with options can only be in one place when options are present, this position is paramaterised.
    def strip_blue(self, pixely, pixelx1, pixelx2, frame):
        count_blue = 1
        count_other = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            if col == ChaseAnalyser.QBLUE:
                count_blue += 1
            else:
                count_other += 1
        if (count_blue / count_other) > 0.3:
            return True
        return False

    def strip_option_blue(self, pixely, pixelx1, pixelx2, frame, threshold):
        count_blue = 1
        count_other = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            if col == ChaseAnalyser.LBLUE:
                count_blue += 1
            else:
                count_other += 1
        if (count_blue / count_other) > 0.2:
            return True

    def strip_green(self, pixely, pixelx1, pixelx2, frame, threshold):
        # frame = cv2.imread(frame)
        count_green = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            # print(frame[pixely][x])
            if col == ChaseAnalyser.GREEN:
                count_green += 1
        # print(count_green)
        if count_green >= threshold:
            return True
        return False

    def strip_mboxgreen(self, pixely, pixelx1, pixelx2, frame, threshold):
        # frame = cv2.imread(frame)
        count_green = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            # print(frame[pixely][x])
            if col == ChaseAnalyser.GREEN:
                count_green += 1
        # print(count_green)
        if count_green >= threshold:
            return True


    def area(self, pixely1, pixelx1, pixelx2, offset, frame, colour):
        for y in range(pixely1, (pixely1+offset), 2):
            if colour == ChaseAnalyser.GREEN or colour == ChaseAnalyser.GREEN:
                strip1 = self.strip_green(y, pixelx1, pixelx2, frame, 0.3)
            else:
                strip1 = self.strip_blue(y, pixelx1, pixelx2, frame)
            # Return boolean of whether colour is found
            return strip1

    def masking(self, img_path):
        # read in image
        # path = 'tests/sc7.png'
        nm = 'sc7'
        img = cv2.imread(img_path)

        # convert to rgb as im.. functions use bgr
        rgb_img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)

        # set range for blue detection
        light_blue = (90, 70, 50)
        dark_blue = (128, 255, 255)
        # create mask for blue detection using bitwise AND
        mask_blue = cv2.inRange(rgb_img, light_blue, dark_blue)
        result_blue = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_blue)

        # make green mask and apply using bitwise AND
        green = (40, 40, 40)
        dark_green = (70, 255, 255)
        mask_green = cv2.inRange(rgb_img, green, dark_green)
        result_green = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_green)

        # make red mask and apply using bitwise AND
        lower_red = np.array([139, 0, 0])
        upper_red = np.array([255, 204, 203])
        mask_red = cv2.inRange(rgb_img, lower_red, upper_red)
        result_red = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_red)

        # make white mask and get everything outside of it
        lower_white = np.array([0, 0, 168])
        upper_white = np.array([172, 111, 255])
        mask_white = cv2.inRange(rgb_img, lower_white, upper_white)
        result_white = cv2.bitwise_not(rgb_img, rgb_img, mask=mask_white)

        # read masks
        mask1 = result_blue
        mask2 = result_white
        mask3 = result_red
        mask4 = result_green

        # add masks
        result = (mask1 + mask3 + mask4)
        result = result.clip(0, 255).astype("uint8")

        # save results
        cv2.imwrite(F"mask{img_path}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    def colour_checker(self, pixel):
        if 0 <= pixel[0] <= 50 and 100 <= pixel[1] <= 233 and 0 <= pixel[2] <= 20:
            return ChaseAnalyser.GREEN
        elif 139 <= pixel[0] <= 256 and 0 <= pixel[1] <= 80 and 0 <= pixel[2] <= 83:
            return ChaseAnalyser.RED
        elif 42 <= pixel[0] <= 70 and 75 <= pixel[1] <= 112 and 105 <= pixel[2] <= 132:
            return ChaseAnalyser.DBLUE
        elif 1 <= pixel[0] <= 32 and 71 <= pixel[1] <= 115 and 140 <= pixel[2] <= 256:
            return ChaseAnalyser.LBLUE
        elif 90 <= pixel[0] <= 130 and 120 <= pixel[1] <= 150 and 180 <= pixel[2] <= 210:
            return ChaseAnalyser.QBLUE
        else:
            return ChaseAnalyser.BLACK

    def logic(self, img_path):
        # analysis = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        analysis = img_path

        # Specify the Y coordinate of the line of pixels we wish to check.
        check = analysis[650]
        count = 0

        seq = []
        current = []
        x_start = 230
        x_end = 1060
        for index, item in islice(enumerate(check), x_start, x_end):
            # skips black pixels
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                continue
            else:
                # Get the colour of the current pixel.
                temp = self.colour_checker(item)
                # If the colour is the same as the previous or black, then skip.
                if temp == current or temp == ChaseAnalyser.BLACK or temp == ChaseAnalyser.QBLUE or temp == ChaseAnalyser.OPTIONGREEN or temp == ChaseAnalyser.OPTIONBLUE:
                    continue
                # Otherwise, if the colour is unique then add it to the sequence,
                # as we only care about new colours marking a new box.
                else:
                    current = temp
                    seq.append(current)
        print(seq)
        return seq

    def inference(self, seq):
        contestant_correct = False
        chaser_correct = False
        # Find if Contestant got it right
        # If LBLUE is present then contestant selected the incorrect answer.
        if ChaseAnalyser.LBLUE not in seq:
            contestant_correct = True
        else:
            contestant_correct = False
        # Find if Chaser got it right
        if len(seq) > 3:
            # If chaser selected middle there will be two red bars around the middle choice.
            if seq.count(ChaseAnalyser.RED) == 2:
                first_red = seq.index(ChaseAnalyser.RED)
                # Check if middle is green
                if seq[first_red + 1] == ChaseAnalyser.GREEN:
                    chaser_correct = True
            # Chaser selected edge
            else:
                # Check if Chaser correctly guessed choice A.
                if seq[0] == ChaseAnalyser.GREEN and seq[1] == ChaseAnalyser.RED:
                    chaser_correct = True
                # Check if Chaser correctly guessed choice C.
                elif seq[-1] == ChaseAnalyser.GREEN and seq[-2] == ChaseAnalyser.GREEN:
                    chaser_correct = True

        # If the length is 3 then it must be both chaser and contestant correct.
        # Green + Red + DBlue  || DBlue + Red + Green
        else:
            chaser_correct = True

        return contestant_correct, chaser_correct


class NumberAnalyser:

    # boilerplate code to pre-process image
    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def thresholding(self, image):
        gray = self.get_grayscale(image)
        (T, threshInv) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # visualize only the masked regions in the image
        masked = cv2.bitwise_not(gray, gray, mask=threshInv)
        return threshInv

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

    # template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    def crop(self, img):
        return img[260:470, 440:830]

    def numbers(self, img):

        # reader = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        pt.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/4.1.3/bin/tesseract'

        img = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = self.crop(img)
        msk = cv2.inRange(img, array([0, 0, 0]), array([179, 180, 255]))  # for high resolution
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dlt = cv2.dilate(msk, krn, iterations=1)
        thr = 255 - cv2.bitwise_and(dlt, msk)
        txt = pt.image_to_string(thr, config='--psm 11, -c tessedit_char_whitelist=0123456789')
        print(txt)
        # cv2.imshow("", msk)
        # cv2.waitKey(0)
        # print('yes')
        # print(pt.image_to_string(reader, config='--psm 11, -c tessedit_char_whitelist=$0123456789'))

    def get_choice(self, img):
        x = ChaseAnalyser
        check = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        mid1 = check[900][1120]
        # print(mid1)
        mid2 = check[900][1775]
        top1 = check[900][1775]
        top2 = check[900][1775]
        low1 = check[900][1120]
        low2 = check[900][1775]
        if x.colour_checker(self, mid1) == ChaseAnalyser.GREEN or x.colour_checker(self, mid2) == ChaseAnalyser.GREEN:
            return (ChaseAnalyser.MidChoice)
        elif x.colour_checker(self, top1) == ChaseAnalyser.GREEN or x.colour_checker(self, top2) == ChaseAnalyser.GREEN:
            return ChaseAnalyser.TopChoice
        elif x.colour_checker(self, low1) == ChaseAnalyser.GREEN or x.colour_checker(self, low2) == ChaseAnalyser.GREEN:
            return ChaseAnalyser.LowChoice
        else:
            print('OH NO')


class TestClass:

    def test(self, test_file):
        chase_analyser = ChaseAnalyser()
        success = 0
        fail = 0
        with open(test_file, 'r') as read_obj:
            # Create a csv.reader object from the input file object
            csv_reader = reader(read_obj)
            next(csv_reader, None)
            for row in csv_reader:
                expected = row[3]
                got = str(chase_analyser.logic(row[2]))
                if expected == got:
                    print('success')
                    success += 1
                else:
                    print('fail ' + row[2])
                    print('expected ' + str(expected))
                    print('but got: ' + str(got))
                    fail += 1
            print(success, fail)


if __name__ == "__main__":
    x = ChaseAnalyser()
    x.get_frames('S5E116.mp4')
    # frame = cv2.imread('24870.png')
    # print(x.strip_green(640, 230, 1060, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.3))
    y = NumberAnalyser()
    # img = cv2.imread('all_options24926.png')
    # y.numbers(img)
