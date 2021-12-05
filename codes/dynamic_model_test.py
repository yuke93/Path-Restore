import cv2
import numpy as np
import os
from dynamic_model import DynamicModel
from util import get_mse_psnr
from util import whole2patch, patch2whole


class DynamicModelTest(DynamicModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_whole_image(self):
        """ test large images """
        # parameters
        test_set = self.kwargs['test_set']
        load_dir_test = self.kwargs['load_dir_test']
        data_size = self.kwargs['data_size']
        stride = self.kwargs['data_stride']
        is_save = self.kwargs['is_save']
        save_folder = self.kwargs['save_folder']
        limit_batch = self.kwargs['limit_batch']

        is_label = True
        if test_set == 'mine':
            noise_levels = ['_mine']
            is_label = False
        else:
            raise NotImplementedError(test_set)
        ex_str = 'ex359_random'  # only when is_save=True

        # load model
        self.saver_test.restore(self.sess_dyn, load_dir_test)

        avg_psnr = 0.
        avg_rb = 0.
        for noi_level in noise_levels:
            # save folder
            if is_save:
                save_folder_cur = save_folder + \
                    'noise{}/{}/'.format(noi_level, ex_str)
                if not os.path.exists(save_folder_cur):
                    os.makedirs(save_folder_cur)

            # get test data
            test_data, test_label = self.dataset.get_test_data(noi_level)

            # test for each image
            psnr_list = []
            # ssim_list = []
            count = 0
            total_rb = 0
            total_patches = 0
            for key in test_label.keys():
                count += 1
                im_data = test_data[key]
                if is_label:
                    im_label = test_label[key]

                # run
                patch_in, pos, count_map = whole2patch(im_data, (data_size, data_size), (stride, stride))
                feed_dict = {self.input_ph_dyn: patch_in, self.is_train_ph: False}
                patch_len = len(patch_in)
                if limit_batch < 0 or limit_batch >= patch_in:
                    patch_out, *actions = self.sess_dyn.run([self.output_dyn] + self.test_info[:-1], feed_dict=feed_dict)
                else:
                    assert 0 < limit_batch < patch_len
                    patch_out = []
                    actions_batch = []
                    # test a batch of patches every time
                    for idx in range(0, patch_len, limit_batch):
                        patch_cur = patch_in[idx: min(idx + limit_batch, patch_len), ...]
                        feed_dict = {self.input_ph_dyn: patch_cur, self.is_train_ph: False}
                        patch_out_cur, *actions_cur = self.sess_dyn.run([self.output_dyn] + self.test_info[:-1], feed_dict=feed_dict)
                        patch_out.append(patch_out_cur)
                        actions_batch.append(actions_cur)
                    patch_out = np.concatenate(patch_out, axis=0)
                    actions = []
                    for act in zip(*actions_batch):
                        act_cur = np.concatenate(act, axis=0)  # num_of_patches
                        actions.append(act_cur)
                im_output = patch2whole(patch_out, pos, count_map, (stride, stride))
                actions = np.array(actions)  # rb_num x num_of_patches
                actions = np.sum(np.clip(actions, 0., 1.), axis=0)
                total_rb += np.sum(actions)
                total_patches += patch_len

                # save image
                if is_save:
                    cv2.imwrite(save_folder_cur + key + '_' + ex_str + '.png', im_output[:, :, ::-1] * 255)

                # PSNR
                if is_label:
                    _, psnr = get_mse_psnr(im_output, im_label)
                    psnr_list.append(psnr)

            if is_label:
                psnr_mean = np.array(psnr_list).mean()
            rb_mean = 1. * total_rb / total_patches
            avg_psnr += psnr_mean
            avg_rb += rb_mean

        avg_rb /= len(noise_levels)
        if is_label:
            avg_psnr /= len(noise_levels)
            print('average PSNR: {:.3f}, average RB: {:.3f}'.format(avg_psnr, avg_rb))
        else:
            print('average RB: {:.3f}'.format(avg_rb))
