import cv2
from PIL import Image, ImageSequence
from skimage import io
import os
import shutil
import subprocess


def cut_width(t, page, page_num, is_png=False):
    '''
    Cut uncessery scanned page from left and right
    Note: there is a difference between pages
    '''
    width, height = page.size
    bottom = height
    right = width
    top = 0
    left = 0
    if page_num == 1:
        left = 150
        bottom = height - 100
        right = width - 50
    elif page_num == 2:
        right = width - 100
        bottom = bottom - 300

    cropped = page.crop((left, top, right, bottom))
    print("###" + str(cropped.size))
    path = t[11:]
    cropped.save('after_cut/' + path)
    return cropped


def get_concat_vertical(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


# def tiff_to_jpeg(tiff):
#     page_count = 0
#     while 1:
#         try:
#             save_name = 'page' + str(page_count) + ".jpeg"
#             try:
#                 tiff.save('temp/' + save_name)
#             except OSError:
#                 '''
#                 Deal with OSError by saving tiff as jpeg
#                 '''
#                 tiff = tiff.convert("RGB")
#                 tiff.save('temp/' + save_name)
#             page_count = page_count + 1
#             tiff.seek(page_count)
#         except EOFError:
#             page1 = Image.open('temp/page0.jpeg')
#             page1 = cut_width(page1, 1)
#             if page_count > 1:
#                 # doc has 2 pages
#                 page2 = Image.open('temp/page1.jpeg')
#                 page2 = cut_width(page2, 2)
#                 concat = get_concat_vertical(page1, page2)
#                 return concat
#             else:
#                 # doc has 1 page
#                 return page1


# def image_to_png(png):
#     width, height = png.size
#     page1, page2 = png.crop((0, 0, width, height / 2)), png.crop((0, height / 2, width, height))
#     page1, page2 = cut_width(page1, 1, True), cut_width(page2, 2, True)
#     concat = get_concat_vertical(page1, page2)
#     return concat


# def get_prepared_doc(name):
#     extantion = name.split('.')[-1]
#     try:
#         img = Image.open(name)
#     except FileNotFoundError:
#         print("ERROR: {}: file not found".format(name))
#         raise
#     if extantion == 'tiff' or extantion == 'tif':
#         image = tiff_to_jpeg(img)
#     if extantion == 'png' or extantion == 'jpeg':
#         image = image_to_png(img)
#     return image


if __name__ == "__main__":
    entries = os.listdir('check/')
    a_or_b = 'a'
    num_page = 1
    for entry in entries:
        print(num_page)
        if num_page == 17:
            num_page += 1
            continue
        img = Image.open('check/' + str(entry))
        array_images = []
        for i in range(2):
            try:
                img.seek(i)
                img.save('result/page_%s.tif' % (i,))
                path = 'page_%s.tif' % (i,)
                base_path = "result/"
                new_path = "after_jpeg/"
                read = cv2.imread('result/page_%s.tif' % (i,))
                outfile = path.split('.')[0] + '.jpeg'
                cv2.imwrite(new_path + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
                page1 = Image.open(new_path + outfile)
                page1 = cut_width(str(new_path + outfile), page1, i + 1, True)
                array_images.append(page1)
            except EOFError:
                break
        page1 = Image.open('after_cut/page_0.jpeg')
        page2 = Image.open('after_cut/page_1.jpeg')
        concat = get_concat_vertical(page1, page2)
        concat.save('final_solution/page_' + str(num_page) + '-' + str(a_or_b) + '.jpeg')
        if a_or_b == 'a':
            a_or_b = 'b'
        else:
            a_or_b = 'a'
            num_page += 1



# another option
"""a_or_b = ['a', 'b']
    for j in range(1, 30):
        for k in a_or_b:
            if j == 17 and k == 'a':
                continue
            img = Image.open('check/' + str(j) + '-' + str(k) + '.tiff')
            array_images = []
            for i in range(2):
                try:
                    img.seek(i)
                    img.save('result/page_%s.tif' % (i,))
                    path = 'page_%s.tif' % (i,)
                    base_path = "result/"
                    new_path = "after_jpeg/"
                    read = cv2.imread('result/page_%s.tif' % (i,))
                    outfile = path.split('.')[0] + '.jpeg'
                    cv2.imwrite(new_path + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
                    page1 = Image.open(new_path + outfile)
                    page1 = cut_width(str(new_path + outfile), page1, i+1, True)
                    array_images.append(page1)
                except EOFError:
                    break
            page1 = Image.open('after_cut/page_0.jpeg')
            page2 = Image.open('after_cut/page_1.jpeg')
            concat = get_concat_vertical(page1, page2)
            concat.save('final_solution/page_' + str(j) + '-' + str(k) + '.jpeg')"""










