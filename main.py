from ex_fwdproj_ans import create_sinogram
from ex_bckproj_ans import reconstruct_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':

    # 画像の読み込み(&表示)
    ref = np.loadtxt('./img/reference.txt')
    plt.imshow(ref, cmap=cm.Greys_r)

    # 各種パラメータの設定
    n_img = ref.shape[0] # 画像サイズ
    n_proj = 36 # 投影数
    n_det = 200 # 検出器の数
    sod, sdd = 300, 600 #　sod:線源とオブジェクト中心の距離, sdd: 線源と検出器の距離

    # Call the function
    sino = create_sinogram(ref, n_img, n_proj, n_det, sod, sdd)

    # サイノグラムの表示
    plt.figure()
    plt.imshow(sino, cmap=cm.Greys_r)
    plt.show()

    n_iter = 100 # 逆投影の反復回数
    lambda_ = 0.5 # 逐次近似法のパラメータ
    subset_size = 36 # 逐次近似法のサブセットサイズ
    prior = "fbp" # 初期画像の設定。fbp: FBPによる初期画像, none: 0で初期化
    mode = "sirt" # 再構成手法。"sirt" or "sart" or "mlem"

    # 逐次近似法による再構成
    if prior == "none":
        rec = np.zeros((n_img, n_img))
    elif prior == "fbp":
        rec = reconstruct_image(n_proj, n_det, n_img, sod, sdd, sino, filter_mode="ram-lak")


    # randomize the order of the angles
    angle_list_all = np.arange(n_proj)
    np.random.shuffle(angle_list_all)

    # make subset
    subset_changed = False
    while n_proj % subset_size != 0:
        subset_size -= 1
        subset_changed = True
    if subset_changed:
        print("Subset size changed to: ", subset_size)

    subset = []
    for i in range(n_proj // subset_size):
        subset.append(angle_list_all[i * subset_size: (i + 1) * subset_size])
    print("Subset: ", subset)

    if mode == "mlem":
        rec = np.ones_like(rec) * 0.01

    for i in range(n_iter):
        print("Iteration: ", i)
        if mode == "sirt":
            sino_tmp = create_sinogram(rec, n_img, n_proj, n_det, sod, sdd)
            diff = sino - sino_tmp
            rec += lambda_ * reconstruct_image(n_proj, n_det, n_img, sod, sdd, diff, filter_mode="none")
        elif mode == "sart":
            for j in range(len(subset)):
                sino_tmp = create_sinogram(rec, n_img, n_proj, n_det, sod, sdd, angle_list=subset[j])
                diff = sino[subset[j], :] - sino_tmp
                rec += lambda_ * reconstruct_image(n_proj, n_det, n_img, sod, sdd, diff, filter_mode="none", angle_list=subset[j])
        elif mode == "mlem":
            for j in range(len(subset)):
                sino_tmp = create_sinogram(rec, n_img, n_proj, n_det, sod, sdd, angle_list=subset[j])
                ratio = sino[subset[j], :] / sino_tmp
                rec *= reconstruct_image(n_proj, n_det, n_img, sod, sdd, ratio, filter_mode="none", angle_list=subset[j])

    # 画像の表示
    plt.figure()
    plt.imshow(rec, cmap=cm.Greys_r)
    plt.show()

    # 画像の保存
    imgname = './img/reconstructed_sirt.txt'
    np.savetxt(imgname, rec)

     
