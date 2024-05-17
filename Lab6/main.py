from PIL import Image
import numpy as np


def correct_wm(img, wm):
    wm = np.tile(wm, (img.shape[0] // wm.shape[0] + 1, img.shape[1] // wm.shape[1] + 1))
    wm = wm[:img.shape[0], :img.shape[1]]
    return wm


def insert_watermark(image, wm, bin_field: int, base=8):
    if(image.shape != wm.shape):
        wm = correct_wm(image, wm)

    image_fields = np.unpackbits(image).reshape((image.shape[0], image.shape[1], base))
    wm_image_fields = image_fields.copy()
    for i in range(image_fields.shape[0]):
        for j in range(image_fields.shape[1]):
            wm_image_fields[i][j][base-bin_field] = image_fields[i][j][base-bin_field] ^ wm[i][j]
    packed_wm_im = np.packbits(wm_image_fields).reshape(image.shape)
    return packed_wm_im

def text_to_8bit(text):
    binary_text = ""
    for char in text:
        ascii_value = ord(char)
        binary_char = format(ascii_value, '08b')
        binary_text += binary_char
    return [i for i in binary_text]


def insert_text(image, text, bin_field: int, base=8):
    text = np.array(text_to_8bit(text))
    text = np.tile(text, (image.shape[0], image.shape[1]))
    image_fields = np.unpackbits(image).reshape((image.shape[0], image.shape[1], base))
    wm_image_fields = image_fields.copy()
    for i in range(image_fields.shape[0]):
        for j in range(image_fields.shape[1]):
            wm_image_fields[i][j][base-bin_field] = image_fields[i][j][base-bin_field] ^ int(text[i][j])
    packed_wm_im = np.packbits(wm_image_fields).reshape(image.shape)
    return packed_wm_im

def get_text(enc_image, image, text_len, bin_field: int, base=8):
    image_fields = np.unpackbits(image).reshape((image.shape[0], image.shape[1], base))
    enc_image_fields = np.unpackbits(enc_image).reshape((enc_image.shape[0], enc_image.shape[1], base))
    text = ''
    for i in range(image_fields.shape[0]):
        for j in range(image_fields.shape[1]):
            text += str(image_fields[i][j][base-bin_field] ^ enc_image_fields[i][j][base-bin_field])
    text = text[:text_len*base]
    orig_text = ''
    for i in range(0, len(text), 8):
        byte = text[i:i+8]
        orig_text += chr(int(byte, 2))

    return orig_text

# img = Image.open('Lab6\imgs\image.png')
# img_b = np.array(img)[:,:,2] # blue chanell
# img.show()
# wm = Image.open('Lab6\imgs\watermark.png').convert("L")
# wm_bin = np.where(np.array(wm) < 128, 1, 0)
# Image.fromarray(np.where(wm_bin==1, 255, 0)).show()

# for i in range(1, 9):
#     cont_b = insert_watermark(img_b, wm_bin, i)
#     res = np.array(img)
#     res[:,:,2] = cont_b
#     Image.fromarray(res).show()


# img = Image.open('Lab6\imgs\image_undersized.png')
# img_b = np.array(img)[:,:,2] # blue chanell
# img.show()
# wm = Image.open('Lab6\imgs\watermark.png').convert("L")
# wm_bin = np.where(np.array(wm) < 128, 1, 0)
# Image.fromarray(np.where(wm_bin==1, 255, 0)).show()

# for i in range(1, 9):
#     cont_b = insert_watermark(img_b, wm_bin, i)
#     res = np.array(img)
#     res[:,:,2] = cont_b
#     Image.fromarray(res).show()



#text encription
img = Image.open('imgs/image.png')
img_b = np.array(img)[:,:,2] # blue chanell

text = 'texttexttexttexttext'

cont_b = insert_text(img_b, text, 8)
res = np.array(img)
res[:, :, 2] = cont_b
# fig, axes = plt.subplots(1,1, figsize=(4, 4))
# axes.imshow(res)

#text extraction
img = Image.open('imgs/image.png')
img_b = np.array(img)[:,:,2] # blue chanell

text = get_text(res[:,:,2], img_b, len(text), 8)
print(text)