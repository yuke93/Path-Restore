import cv2
import os


class MyData(object):

    def __init__(self, **kwargs):
        self.is_train = kwargs['is_train']
        self.test_set = kwargs['test_set']

        # test data
        if not self.is_train:
            self._test_data = None
            if self.test_set == 'mine':
                image_path = '../data/test/mine/'
                self._test_data = self.generate_test_image(image_path)

                # dummy label (no GT)
                self._test_label = self._test_data

            else:
                raise NotImplementedError(self.test_set)
        else:
            raise NotImplementedError('Training data is not implemented.')

    def generate_test_image(self, image_path):
        """ generate test images
        :param image_path: path of test images
        :return: test images
        """
        files = os.listdir(image_path)
        files = [
            file for file in files if
            file.lower().endswith('png') or
            file.lower().endswith('jpg') or
            file.lower().endswith('jpeg')
        ]
        test_image = {}
        for file in files:
            img = cv2.imread(image_path + file)[:, :, ::-1] / 255.
            filename, _ = os.path.splitext(file)
            test_image[filename] = img
        return test_image

    def get_test_data(self):
        return self._test_data, self._test_label
