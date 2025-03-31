#　練習：
#   断面画像(reference.txt)から、サイノグラムを作成する。
#   画像は、128x128のサイズで、投影数は180、検出器の数は200とする。

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def back_projection(n_pix, attenuation, start_point, angle_rad):
    """
    サイノグラムから逆投影を行い、画像を再構成する。
    
    Args:
    sinogram: サイノグラム
    n_iter: 逆投影の反復回数

    Returns:
    image: 再構成画像
    """
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # 逆投影後の画像
    rec = np.zeros((n_pix, n_pix))
    
    # 線源の位置
    x, y = start_point

    # (演習) attenuationをrayに沿って逆投影する

    return rec
    

def rotate(point, origin, angle):
    """
    originを中心に、pointをangle分回転させる。
    
    Args:
    point: 回転させる点 (x, y座標)
    origin: 回転の中心
    angle: 回転角度（radian）

    Returns:
    rotated_points: 回転後の点 (x, y座標)
    """

    # rotate points around origin
    s = np.sin(angle)
    c = np.cos(angle)
    ox, oy = origin

    
    rot_mat = np.array([
        [c, -s, ox - ox * c + oy * s],
        [s, c, oy - ox * s - oy * c],
        [0, 0, 1]
    ], dtype=np.float32)


    points = np.concatenate([point, np.ones(1)], axis=0) # add 1 for translation

    rotated_points = np.matmul(rot_mat, points.T).T # rotate points

    return rotated_points[:2]

def _filter_fn(x, filter_mode="ram-lak", cutoff=0.5):
    if filter_mode == "ram-lak":
        res = 1
        # (演習２) ram-lak filterを実装（周波数領域ではRamp filterに相当）
        # 周波数の絶対値を取り、cutoffより大きい周波数成分を0にする
        # Hint: np.whereを使う
    elif filter_mode == "none":
        res = 1
    else:
        raise ValueError("Unknown filter mode")
    return res

def apply_filter(proj, filter_mode="ram-lak"):

    n_proj = proj.shape[0]
    n_det = proj.shape[1]

    proj_filtered = np.zeros_like(proj)

    print("filter_mode: ", filter_mode)

    for i in range(n_proj):
        
        # Ram-Lak filter
        fft_ = np.fft.fft(proj[i,:]) # FFT (y-axis)
        freq = np.fft.fftfreq(n_det) # frequency (x-axis)
        
        filter_ = _filter_fn(freq, filter_mode) # apply filter
        proj_filtered[i,:] = np.fft.ifft(fft_ * filter_).real

    return proj_filtered


if __name__ == '__main__':

    # サイノグラム読み込み
    sino = np.loadtxt('./img/sinogram.txt')

    # 各種パラメータの設定
    n_proj, n_det = sino.shape # サイノグラムサイズ
    n_iter = 1 # 逆投影の反復回数
    n_img = 128 # 画像サイズ(一辺の長さ)
    #n_det = 200 # 検出器の数
    sod, sdd = 300, 600 #　sod:線源とオブジェクト中心の距離, sdd: 線源と検出器の距離

    theta_list = np.linspace(0, 2 * np.pi, n_proj) # 投影角度のリスト
    phi_list = np.arctan(np.linspace(-n_det / 2, n_det / 2, n_det) / sdd) # 検出器の各ピクセルの角度

    filter_mode = "none"  # フィルターの種類
    #filter_mode = "ram-lak"  # フィルターの種類

    # 再構成画像の作成
    rec = np.zeros((n_img, n_img)) # サイノグラムは、投影数x検出器数の画像

    # ray-castingの要領で、各投影を計算
    tmp_x = n_img / 2 - sod
    tmp_y = n_img / 2
    start_point_orig = np.array([tmp_x, tmp_y])

    # フィルター
    sino_filtered = apply_filter(sino, filter_mode)

    # 逆投影
    for i in range(n_proj):
        start_point = rotate(start_point_orig, np.array([n_img / 2, n_img / 2]), theta_list[i])
        for j in range(n_det):
            rec += back_projection(n_img, sino_filtered[i,j], start_point, theta_list[i] + phi_list[j], debug=False)

    # スケーリング
    rec /= n_proj

    # 投影像の表示
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(rec, cmap=cm.Greys_r)
    plt.subplot(1,2,2)
    plt.imshow(np.loadtxt('./img/reference.txt'), cmap=cm.Greys_r)
    plt.show()

    # 画像の保存
    imgname = './img/reconstructed_' + filter_mode + '.txt'
    np.savetxt(imgname, rec)





