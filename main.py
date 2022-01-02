from pytube import YouTube
import sys
import cv2
import numpy as np
from itertools import islice
from csv import reader
import pytesseract as pt


class ChaseAnalyser():
    GREEN = 'green'
    LBLUE = 'lightblue'
    DBLUE = 'darkblue'
    RED = 'red'
    BLACK = 'black'
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

    def download_videos(self):
        with open(sys.argv[1]) as f:
            lines = f.readlines()
        for url in lines:
            yt = YouTube(url)

            streams = yt.streams.filter(progressive=True)
            itag = streams[0].itag
            stream = yt.streams.get_by_itag(itag)
            stream.download()

    def masking(self):
        # read in image
        path = 'tests/sc7.png'
        nm = 'sc7'
        img = cv2.imread(path)

        # convert to rgb as im.. functions use bgr
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        #
        # cv2.imshow('result', result_white)
        # cv2.waitKey(0)

        # read masks
        mask1 = result_blue
        mask2 = result_white
        mask3 = result_red
        mask4 = result_green

        # add masks
        result = (mask1 + mask3 + mask4)
        result = result.clip(0, 255).astype("uint8")

        # save results
        cv2.imwrite(F"mask{nm}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        # show results
        # cv2.imshow('result', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def colourChecker(self, pixel):
        if 51 <= pixel[0] <= 122 and 156 <= pixel[1] <= 233 and 0 <= pixel[2] <= 109:
            return ChaseAnalyser.GREEN
        elif 139 <= pixel[0] <= 149 and 50 <= pixel[1] <= 80 and 50 <= pixel[2] <= 83:
            return ChaseAnalyser.RED
        elif 42 <= pixel[0] <= 70 and 75 <= pixel[1] <= 112 and 105 <= pixel[2] <= 132:
            return ChaseAnalyser.DBLUE
        elif 128 <= pixel[0] <= 132 and 155 <= pixel[1] <= 208 and 200 <= pixel[2] <= 239:
            return ChaseAnalyser.LBLUE
        else:
            return ChaseAnalyser.BLACK

    def logic(self, img_path):
        analysis = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        check = analysis[1370]
        seq = []
        current = []
        for index, item in islice(enumerate(check), 900, 2163):
            # skips black pixels
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                continue
            else:
                temp = self.colourChecker(item)
                if temp == current or temp == ChaseAnalyser.BLACK:
                    continue
                else:
                    current = temp
                    seq.append(current)
        #     print(index)
        print(seq)
        return seq

    def inference(self, seq):
        temp = []
        final = []
        chaser_right = ChaseAnalyser.Placeholder
        contestant_right = ChaseAnalyser.Placeholder
        right_answer = ChaseAnalyser.Placeholder

        for i in seq:
            if i == ChaseAnalyser.RED:
                continue
            else:
                temp.append(i)
        # print(temp)

        if len(temp) == 2:
            contestant_right = ChaseAnalyser.ContestantRight
            if temp[0] == ChaseAnalyser.GREEN:
                right_answer = ChaseAnalyser.ARight
                # print('A')
                final.append(contestant_right)
            else:
                right_answer = ChaseAnalyser.CRight
                # print('B')
                final.append(contestant_right)

        elif len(temp) == 3:
            contestant_right = ChaseAnalyser.ContestantWrong
            if temp[0] == ChaseAnalyser.DBLUE and temp[1] == ChaseAnalyser.DBLUE or temp[1] == ChaseAnalyser.DBLUE and temp[2] == ChaseAnalyser.DBLUE:
                contestant_right = ChaseAnalyser.ContestantRight
                final.append(contestant_right)
            else:
                if temp[0] == ChaseAnalyser.GREEN:
                    right_answer = ChaseAnalyser.ARight
                    print('C')
                elif temp[1] == ChaseAnalyser.GREEN:
                    print('D')
                    right_answer = ChaseAnalyser.BRight
                    if temp[0] == ChaseAnalyser.DBLUE and temp[2] == ChaseAnalyser.DBLUE:
                        contestant_right = ChaseAnalyser.ContestantRight
                else:
                    right_answer = ChaseAnalyser.CRight
                    contestant_right = ChaseAnalyser.ContestantWrong
                    print('E')
                final.append(contestant_right)
        else:
            print('Something is terribly wrong..')

        reds = 0
        for i in seq:
            if i == ChaseAnalyser.RED:
                reds += 1
        if reds == 2 and ChaseAnalyser.BRight:
            chaser_right = ChaseAnalyser.ChaserRight
            final.append(chaser_right)
        elif reds == 1:
            if right_answer == ChaseAnalyser.ARight and seq[1] == ChaseAnalyser.RED:
                chaser_right = ChaseAnalyser.ChaserRight
            elif right_answer == ChaseAnalyser.CRight and seq[-2] == ChaseAnalyser.RED:
                chaser_right = ChaseAnalyser.ChaserRight
            else:
                chaser_right = ChaseAnalyser.ChaserWrong
            final.append(chaser_right)

        print(final)

    # def get_frames(self, stream):
    #     cap = cv2.VideoCapture(stream)
    #     i = 0
    #     while (cap.isOpened()):
    #         ret, frame = cap.read()
    #         if ret == False:
    #             break
    #         if i % 10 == 0:
    #             cv2.imwrite('kang' + str(i) + '.jpg', frame)
    #         i += 1
    #
    #     cap.release()
    #     cv2.destroyAllWindows()


class NumberAnalyser:

    def numbers(self, img_path):
        reader = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(pt.image_to_string(reader))


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
    x.masking()
    # x.logic('masksc7.png')
    x.inference(x.logic('masksc19.png'))
    # test = TestClass()
    # test.test('tests/test.txt')
    number_analyser = NumberAnalyser()
    number_analyser.numbers('num1.png')
