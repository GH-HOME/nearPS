import cv2


class rect_drawer(object):
    """
    draw a rect by mouce
    """

    def __init__(self, image):
        assert image is not None
        self.img = image
        self.window_name = 'draw rect'
        cv2.namedWindow(self.window_name)
        self.tlx = 0
        self.tly = 0
        self.rect = None


    def build_mouse(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name,self.img)
        cv2.setMouseCallback(self.window_name, self.onPick)
        cv2.waitKey(0)

    def finish(self):
        cv2.destroyAllWindows()

    def onPick(self, event, x, y, flags, param):
        rectangle = False
        drawimg = self.img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            self.tlx, self.tly = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle == True:
                self.rect = (min(self.tlx, x), min(self.tly, y), abs(self.tlx - x), abs(self.tly - y))


        elif event == cv2.EVENT_LBUTTONUP:
            rectangle = False
            rect_over = True
            cv2.rectangle(drawimg, (self.tlx, self.tly), (x, y), (0, 255, 0), 2)
            self.rect = (min(self.tlx, x), min(self.tly, y), abs(self.tlx - x), abs(self.tly - y))

            cv2.imshow(self.window_name, drawimg)




if __name__ == '__main__':
    path = '../data/col_wise/0D8A4053.JPG'
    myrect = rect_drawer(path)
    myrect.build_mouse()
    myrect.finish()