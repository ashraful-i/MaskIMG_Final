# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd
from glob import glob


# Press the green button in the gutter to run the script.

def calc_prob(d: pd.DataFrame, lst1, col, pname):
    d["count"] = d.groupby(lst1)[col].transform('count')
    d_len = len(d)
    d[pname] = d["count"] / d_len
    return d


def add_trainset_prob_to_testset(test_df: pd.DataFrame, prob: pd.DataFrame, on_l):
    df_merge = pd.merge(test_df, prob, on=on_l, how='left')
    return df_merge


def make_dataset():
    mask_img_path = "F:\\6-Training_Learining\PythonLearn\Data_Mine\ImageDetect\ibtd\Mask"
    img_path = "F:\\6-Training_Learining\PythonLearn\Data_Mine\ImageDetect\ibtd\ibtd"

    mask_img_file_list = glob(mask_img_path + "\*")
    img_file_list = glob(img_path + "\*")
    mask_img_file_list = mask_img_file_list[:-1]
    img_file_list = img_file_list[:-1]
    print(len(mask_img_file_list))
    print(len(img_file_list))

    total_images = len(img_file_list)

    half_len = int(total_images / 2)
    #mask_img_file_list_1 = mask_img_file_list[:half_len]
    #img_file_list_1 = img_file_list[:half_len]
    print(half_len)
    RGB_S = []
    for cnt_img in range(total_images):
        print(cnt_img)
        im_main = Image.open(img_file_list[cnt_img])
        im_mask = Image.open(img_file_list[cnt_img])
        pix_main = im_main.load()
        pix_mask = im_mask.load()
        im_main_w, im_main_h = im_main.size
        for x in range(im_main_w):
            for y in range(im_main_h):
                Blue_mask = pix_mask[x, y][2]
                Green_mask = pix_mask[x, y][1]
                Red_mask = pix_mask[x, y][0]
                # print("B G R "+ str(Blue) +" "+str(Green)+" "+str(Red)+"\n")
                if Blue_mask == 255 and Green_mask == 255 and Red_mask == 255:
                    Blue_main = pix_main[x, y][2]
                    Green_main = pix_main[x, y][1]
                    Red_main = pix_main[x, y][0]
                    rgb_nskin = []
                    rgb_nskin.append(Blue_main)
                    rgb_nskin.append(Green_main)
                    rgb_nskin.append(Red_main)
                    rgb_nskin.append(1)
                    RGB_S.append(rgb_nskin)
                else:
                    rgb_skin = []
                    rgb_skin.append(Blue_mask)
                    rgb_skin.append(Green_mask)
                    rgb_skin.append(Red_mask)
                    rgb_skin.append(2)
                    RGB_S.append(rgb_skin)
        im_main.close()
        im_mask.close()
    df_skin = pd.DataFrame(RGB_S, columns=['B', 'G', 'R', 'Skin'])
    print(df_skin.shape)
    return df_skin


#   sk_db:pd.DataFrame, nsk_db:pd.DataFrame
def checkimage(img_t, sk_db: pd.DataFrame, nsk_db: pd.DataFrame):
    im_main = Image.open(img_t)
    pix_main = im_main.load()
    im_main_w, im_main_h = im_main.size
    for x in range(im_main_w):
        for y in range(im_main_h):
            Blue_mask = pix_main[x, y][2]
            Green_mask = pix_main[x, y][1]
            Red_mask = pix_main[x, y][0]
            # print("B G R " + str(Blue_mask) + " " + str(Green_mask) + " " + str(Red_mask) + "\n")

            cond_sk = (sk_db['B'] == Blue_mask) & (sk_db['G'] == Green_mask) & (sk_db['R'] == Red_mask)
            get_skin_prob = sk_db[cond_sk].skin_prob.values
            cond_nsk = (nsk_db['B'] == Blue_mask) & (nsk_db['G'] == Green_mask) & (nsk_db['R'] == Red_mask)
            get_nskin_prob = nsk_db[cond_nsk].non_skin_prob.values
            print(get_skin_prob)
            print(get_nskin_prob)
            print("--")
            if len(get_skin_prob):
                if len(get_nskin_prob):
                    if get_skin_prob[0] > get_nskin_prob[0]:
                        pix_main[x, y] = (0, 0, 0)
                    else:
                        pass
                else:
                    pix_main[x, y] = (0, 0, 0)
            else:
                pass
    im_main.save("result.jpg")
    im_main.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_s = make_dataset()
    print("Training Done")
    skin_db = df_s.loc[df_s['Skin'] == 2]
    skin_db = pd.DataFrame(skin_db).reset_index(drop=True)
    skin_db = calc_prob(skin_db, ['B', 'G'], "R", "skin_prob")
    skin_db = skin_db.drop_duplicates()
    print(skin_db.head())
    non_skin_db = df_s.loc[df_s['Skin'] == 1]
    non_skin_db = pd.DataFrame(non_skin_db).reset_index(drop=True)
    non_skin_db = calc_prob(non_skin_db, ['B', 'G'], "R", "non_skin_prob")
    non_skin_db = non_skin_db.drop_duplicates()
    print(non_skin_db.head())
    print("check image")
    checkimage('p1.webp', skin_db, non_skin_db)
    # sample_data()
