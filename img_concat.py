from os import listdir, path
from PIL import Image

space_between_row = 10
new_image_path = 'result'
im_dir = './results/rainy_sunny_cyclegan/test_latest/images'


# get sorted list of images
im_path_list_real = [path.join(im_dir, f) for f in sorted(listdir(im_dir)) if 'real_B' in f]
im_path_list_fake = [path.join(im_dir, f) for f in sorted(listdir(im_dir)) if 'fake_A' in f]

print im_path_list_real[:10]
print im_path_list_fake[:10]

# open images and calculate total widths and heights
im_list = []
total_width = 0
total_height = 0
max_width = 0
max_height = 0

real_images = []
for n in im_path_list_real:
    img = Image.open(n)
    real_images.append(img.copy())
    img.close()
widths, heights = zip(*(i.size for i in real_images))
max_height = max(max_height, max(heights))
max_width = max(max_width, max(widths))

fake_images = []
for n in im_path_list_fake:
    img = Image.open(n)
    fake_images.append(img.copy())
    img.close()
widths, heights = zip(*(i.size for i in fake_images))
max_height = max(max_height, max(heights))
max_width = max(max_width, max(widths))

assert len(real_images) == len(fake_images)
n = len(real_images)


imgs_per_row = 8
imgrows_per_result = 10
for ir in range(n/imgs_per_row/imgrows_per_result):
    new_im = Image.new('RGB', (imgs_per_row*max_width, imgrows_per_result*(max_height+space_between_row)))
    y_offset = 0
    for i in range(imgrows_per_result):
        x_offset = 0
        for j in range(imgs_per_row):
            print i, j
            new_im.paste(real_images[ir*imgs_per_row*imgrows_per_result + i*imgs_per_row+j], (x_offset, y_offset))
            x_offset += max_width
            new_im.paste(fake_images[ir*imgs_per_row*imgrows_per_result + i*imgs_per_row+j], (x_offset, y_offset))
            x_offset += max_width

        y_offset += max_height 
        y_offset += space_between_row


    # show and save
    new_im.save(new_image_path+'-'+str(ir)+'.jpg')
