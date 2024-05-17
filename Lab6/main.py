from PIL import Image
import numpy as np


def insert_watermark(image, wm, bin_field: int, base=8):
    image_fields = np.unpackbits(image).reshape((image.shape[0], image.shape[1], base))
    wm_image_fields = image_fields.copy()
    for i in range(image_fields.shape[0]):
        for j in range(image_fields.shape[1]):
            # wm_image_fields[i][j][base-bin_field] = image_fields[i][j][base-bin_field] & wm[i][j]
            wm_image_fields[i][j][base-bin_field] = image_fields[i][j][base-bin_field] ^ wm[i][j]
            # print('wm_image_res_pixel', wm_image_fields[i][j], 'image_f', image_fields[i][j], 'wm', wm[i][j])
    # wm_image[:,:] = image_fields[:,:,bin_field] & wm[:image.shape[0],:image.shape[1]]
    packed_wm_im = np.packbits(wm_image_fields).reshape(image.shape)
    # out = np.where(packed_wm_im == 1, 255, 0)
    return packed_wm_im


img = Image.open('Lab6\imgs\image.png')
img_b = np.array(img)[:,:,2]
img.show()
wm = Image.open('Lab6\imgs\watermark.png').convert("L")
# wm_bin = np.array(wm.point(lambda x: 0 if x < 128 else 1, '1'))
# wm_bin = np.where(wm_bin == True, 1, 0)
wm_bin = np.where(np.array(wm) < 128, 0, 1)
# print('wm_bin:', wm_bin)
Image.fromarray(np.where(wm_bin==1, 255, 0)).show()
# ? gray scale -> color
# img_data = np.array(img)

for i in range(1, 9):
    cont_b = insert_watermark(img_b, wm_bin, i)
    res = np.array(img)
    res[:,:,2] = cont_b
    Image.fromarray(res).show()


print()