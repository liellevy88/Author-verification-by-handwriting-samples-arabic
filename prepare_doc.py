from PIL import Image
import os


def cut_width(page, page_num):
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
    return cropped


def get_concat_vertical(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def prepare_document():
    dirName = 'files_after_concat'  # 'connected files'
    files = os.listdir('final_test/')
    file_number = 1
    i = 0
    for file in files:
        i += 1
        img = Image.open('final_test/' + str(file))
        # file = os.path.splitext(file)[0] + ".jpeg"
        array_images = []
        try:
            page1 = cut_width(img, 1)
            array_images.append(page1)
            img.seek(1)
            page2 = cut_width(img, 2)
            array_images.append(page2)
        except EOFError:
            break

        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        if file_number == 1:
            concat_1 = get_concat_vertical(page1, page2)
            concat_1.save('final_test_files/page_' + str(i) + '.jpeg')
            file_number = 2
        else:
            concat_2 = get_concat_vertical(page1, page2)
            concat_2.save('final_test_files/page_' + str(i) + '.jpeg')


# if __name__ == "__main__":
#     prepare_document()

