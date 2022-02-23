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
    GREEN = 'green'
    OPTIONGREEN = 'money box green'
    OPTIONBLUE = 'money box blue'
    OPTIONBLUE2 = 'middle money box blue'
    OPTIONBLACK = 'money box blue'
    WHITE = 'white'
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
    TopChoice = ', Top choice'
    MidChoice = ', Middle choice'
    LowChoice = ', Low choice'
    ContestantWins = ', Contestant Wins'
    ChaserWins = ', Chaser Wins'
    frames_to_skip = 2
    STARTINGSTATE = 0
    OPTIONSTATE = 1
    QUESTIONSTATE = 2
    buffer = 5

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
        state = ChaseAnalyser.STARTINGSTATE
        array = []
        money_choices = []
        ops = None
        choice = None
        win_loss = None
        streamName = stream + ', '
        current_log = []
        write_data = False
        # Loop through each frame in the video.
        while video.isOpened():
            i += 1
            ret, frame_bgr = video.read()
            if not ret:
                break
            else:

                # Write contestant data to csv
                if write_data:
                    f = open("information.txt", "a")
                    if current_log != []:
                        for data in current_log:
                            f.write(data)
                        #f.write(str(current_log))
                        f.write('\n')
                    f.close()
                    current_log = []
                    write_data = False


                # # Convert the frame into RGB colour format.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # # Skip some frames for efficiency.
                if i % self.frames_to_skip == 0:
                    if state == ChaseAnalyser.STARTINGSTATE:
                        # get options for the money that are presented to the contestant
                        if not self.check_blue_options(frame_rgb):
                            if len(money_choices) == 5:
                                print('money choices', len(money_choices))
                                repeater = True
                                for num, img in enumerate(money_choices):
                                    print('yes')
                                    cv2.imwrite(F"3/moneyarray{num}.png", img)
                                options = cv2.cvtColor(money_choices[0], cv2.COLOR_BGR2RGB)
                                y = NumberAnalyser()
                                current_log.append(streamName)
                                current_log.append(str(y.numbers(options)))
                                state = ChaseAnalyser.OPTIONSTATE
                                print(state)
                                money_choices = []
                                array = []
                                continue
                        elif self.check_blue_options(frame_rgb):
                            all_money = frame_bgr
                            if len(money_choices) < 5:
                                money_choices.append(all_money)
                            else:
                                money_choices.pop(0)
                                money_choices.append(all_money)


                    # get option chosen by contestant
                    if state == ChaseAnalyser.OPTIONSTATE:
                        if not self.check_green_options(frame_rgb):
                            if green_box is not None:
                                state = ChaseAnalyser.QUESTIONSTATE
                                cv2.imwrite(F"3/{i}.png", green_box)
                                options = cv2.cvtColor(green_box, cv2.COLOR_BGR2RGB)
                                if repeater:
                                    if self.strip_mboxgreen(375, 450, 830, options, 70):
                                        array = [5, 8]
                                        choice = ChaseAnalyser.MidChoice
                                        current_log.append(choice)
                                        print(choice)
                                        state = ChaseAnalyser.QUESTIONSTATE
                                        print(state, array)
                                    elif self.strip_mboxgreen(440, 450, 830, options, 70):
                                        array = [4, 8]
                                        choice = ChaseAnalyser.LowChoice
                                        current_log.append(choice)
                                        print(choice)
                                        state = ChaseAnalyser.QUESTIONSTATE
                                        print(state, array)
                                    elif self.strip_mboxgreen(300, 450, 830, options, 70):
                                        array = [6, 8]
                                        choice = ChaseAnalyser.TopChoice
                                        current_log.append(choice)
                                        print(choice)
                                        state = ChaseAnalyser.QUESTIONSTATE
                                        print(state, array)
                                    else:
                                        print('OH NO')
                                green_box = None
                                repeater = False
                                continue
                        elif self.check_green_options(frame_rgb):
                            green_box = frame_bgr

                    if state == ChaseAnalyser.QUESTIONSTATE:
                        # get question boxes once the right answer has  been highlighted green
                        if x.strip_green(650, 230, 1050, frame_rgb, 20):
                            last_frame = frame_bgr
                        # get outline of question (with no options) to signify the show has moved on to next q
                        elif x.strip_blue(590, 230, 1050, frame_rgb) or x.check_blue_options(frame_rgb) or i >= 70000:
                            if last_frame is not None:
                                cv2.imwrite(F"3/questions{i}.png", last_frame)
                                # Convert the frame to RGB to analyse.
                                last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                                # Finds the sequence of colours corresponding to the given answers from Chaser and Contestant.
                                answer_sequence = x.logic(last_frame)
                                # Takes the colour sequence and infers whether the Chaser & Contestant were right or wrong.
                                contestant_correct, chaser_correct = x.inference(answer_sequence)
                                # print('frame ' + str(i))
                                # print(f'Chaser was correct is {chaser_correct}\nContestant is correct is {contestant_correct}')
                                if chaser_correct:
                                    array[1] -= 1
                                if contestant_correct:
                                    array[0] -= 1
                                print(array)
                                if array[0] == 0 and array[1] != 0:
                                    win_loss = ChaseAnalyser.ContestantWins
                                    current_log.append(win_loss)
                                    # current_log.append('\n')
                                    print(win_loss)
                                    state = ChaseAnalyser.STARTINGSTATE
                                    print(state)
                                elif array[0] == array[1]:
                                    win_loss = ChaseAnalyser.ChaserWins
                                    current_log.append(win_loss)
                                    print(win_loss)
                                    state = ChaseAnalyser.STARTINGSTATE
                                    print(state)
                                # Reset to find the next question.
                                last_frame = None
                                write_data = True
                                continue

        video.release()
        cv2.destroyAllWindows()

    def check_blue_options(self, frame_rgb):
        if x.strip_option_blue(300, 450, 830, frame_rgb, 0.3) and x.strip_option_blue(440, 450, 830, frame_rgb, 0.3) and x.strip_option_blue_middle(375, 450, 830, frame_rgb, 0.3):
            return True
        return False

    def check_green_options(self, frame_rgb):
        op1 = x.strip_mboxgreen(300, 450, 830, frame_rgb, 70) #top option
        op2 = x.strip_mboxgreen(375, 450, 830, frame_rgb, 70) #middle option
        op3 = x.strip_mboxgreen(440, 450, 830, frame_rgb, 70) #low option
        if op1 or op2 or op3:
            return True
        # if op1:
        #     if x.black_strip(215, 450, 830, frame_rgb, 70) and x.black_strip(375, 450, 830, frame_rgb, 70):
        #         return True
        # elif op2:
        #     if x.black_strip(215, 450, 830, frame_rgb, 70) and x.black_strip(440, 450, 830, frame_rgb, 70):
        #         return True
        # elif op3:
        #     if x.black_strip(215, 450, 830, frame_rgb, 70) and x.black_strip(500, 450, 830, frame_rgb, 70):
        #         return True
        # else:
        #     return False

    def black_strip(self, pixely, pixelx1, pixelx2, frame, threshold):
        count_black = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            # print(frame[pixely][x])
            if col == ChaseAnalyser.OPTIONBLACK:
                count_black += 1
        # print(count_green)
        if count_black >= threshold:
            return True

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

    def strip_option_blue(self, pixely, pixelx1, pixelx2, frame, threshold):
        count_blue = 1
        count_other = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            if col == ChaseAnalyser.OPTIONBLUE:
                count_blue += 1
            else:
                count_other += 1
        if (count_blue / count_other) > threshold:
            return True
        return False

    def strip_option_blue_middle(self, pixely, pixelx1, pixelx2, frame, threshold):
        count_blue = 1
        count_other = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            if col == ChaseAnalyser.OPTIONBLUE2:
                count_blue += 1
            else:
                count_other += 1
        if (count_blue / count_other) > threshold:
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

    def strip_mboxgreen(self, pixely, pixelx1, pixelx2, frame, threshold):
        # frame = cv2.imread(frame)
        count_green = 1
        for x in range(pixelx1, pixelx2):
            col = self.colour_checker(frame[pixely][x])
            # print(frame[pixely][x])
            if col == ChaseAnalyser.OPTIONGREEN:
                count_green += 1
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
        if 0 <= pixel[0] <= 20 and 150 <= pixel[1] <= 233 and 0 <= pixel[2] <= 20:
            return ChaseAnalyser.GREEN
        elif 139 <= pixel[0] <= 256 and 0 <= pixel[1] <= 80 and 0 <= pixel[2] <= 83:
            return ChaseAnalyser.RED
        elif 30 <= pixel[0] <= 70 and 75 <= pixel[1] <= 112 and 105 <= pixel[2] <= 132:
            return ChaseAnalyser.DBLUE
        elif 1 <= pixel[0] <= 32 and 71 <= pixel[1] <= 130 and 140 <= pixel[2] <= 256:
            return ChaseAnalyser.LBLUE
        elif 120 <= pixel[0] <= 160 and 130 <= pixel[1] <= 170 and 180 <= pixel[2] <= 220:
            return ChaseAnalyser.QBLUE
        elif 30 <= pixel[0] <= 80 and 150 <= pixel[1] <= 240 and 0 <= pixel[2] <= 20:
            return ChaseAnalyser.OPTIONGREEN
        elif 0 <= pixel[0] <= 15 and 89 <= pixel[1] <= 219 and 190 <= pixel[2] <= 256:
            return ChaseAnalyser.OPTIONBLUE
        elif 0 <= pixel[0] <= 10 and 0 <= pixel[1] <= 10 and 100 <= pixel[2] <= 256:
            return ChaseAnalyser.OPTIONBLUE2
        # elif 15 <= pixel[0] <= 25 and 15 <= pixel[1] <= 25 and 20 <= pixel[2] <= 35:
        #     return ChaseAnalyser.OPTIONBLACK
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
                if temp == current or temp == ChaseAnalyser.BLACK or temp == ChaseAnalyser.QBLUE or \
                        temp == ChaseAnalyser.OPTIONGREEN or temp == ChaseAnalyser.OPTIONBLUE or temp == ChaseAnalyser.OPTIONBLACK:
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
                elif seq[-1] == ChaseAnalyser.GREEN and seq[-2] == ChaseAnalyser.RED:
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

    def analyse_options(self, options):
        final = []
        options = options.splitlines()

        for i in options:
            if i == '':
                continue
            else:
                # i = int(i)
                final.append(i)
        print(np.sort(final))
        return np.sort(final)

        # print(options[)
        # for i in options:
        #     for j in options:
        #         if i < j:
        #             i = j
        # print(options)

    def numbers(self, img):
        x = NumberAnalyser()
        # reader = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        pt.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/4.1.3/bin/tesseract'
        img = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = self.crop(img)
        msk = cv2.inRange(img, array([0, 0, 0]), array([179, 180, 255]))  # for high resolution
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dlt = cv2.dilate(msk, krn, iterations=1)
        thr = 255 - cv2.bitwise_and(dlt, msk)
        txt = pt.image_to_string(thr, config='--psm 12, -c tessedit_char_whitelist=$0123456789')
        print(txt)
        ops = x.analyse_options(txt)
        # cv2.imshow("", thr)
        # cv2.waitKey(0)
        # cv2.imshow("", msk)
        # cv2.waitKey(0)
        return ops


        # print('yes')
        # print(pt.image_to_string(reader, config='--psm 11, -c tessedit_char_whitelist=$0123456789'))


    def get_choice(self, img):
        x = ChaseAnalyser
        check = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mid1 = check[375][450]
        mid2 = check[375][830]
        top1 = check[300][450]
        top2 = check[300][830]
        low1 = check[440][450]
        low2 = check[440][830]
        if x.colour_checker(mid1) == x.OPTIONGREEN:
            return ChaseAnalyser.MidChoice
        elif x.strip_mboxgreen(440, 450, 830, check, 70):
            return ChaseAnalyser.LowChoice
        elif x.strip_mboxgreen(300, 450, 830, check, 70):
            return  ChaseAnalyser.TopChoice
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
    # x.download_videos()
    x.get_frames('134.mp4')
    # x.get_frames('142.mp4')
    # x.get_frames('143.mp4')
    # x.get_frames('143.mp4')
    # x.get_frames('144.mp4')
    # x.get_frames('145.mp4')
    # x.get_frames('146.mp4')
    # x.get_frames('147.mp4')
    # x.get_frames('148.mp4')
    # x.get_frames('149.mp4')
    # x.get_frames('150.mp4')
    # x.get_frames('151.mp4')
    # x.get_frames('152.mp4')
    # x.get_frames('153.mp4')
    # x.get_frames('154.mp4')
    # x.get_frames('155.mp4')
    y = NumberAnalyser()
















