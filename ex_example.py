#　練習：
#   断面画像(reference.txt)から、サイノグラムを作成する。
#   画像は、128x128のサイズで、投影数は180、検出器の数は200とする。

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def simple_ray_casting(image, start_point, angle_rad, bound, debug=False):
    """
    2次元画像に対するレイキャスティングを行い、
    指定された起点と角度から見たときの総減衰量を計算する。
    
    Args:
    image: 2次元のNumPy配列（画像）
    start_point: レイの起点(x, y座標)
    angle: レイの角度（度単位）
    bound: 画像の範囲外とみなす距離
    debug: デバッグモードの有無

    Returns:
    attenuation: 総減衰量
    """
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # 画像の幅と高さ
    height, width = image.shape

    history = []
    
    # レイキャスティング
    attenuation = 0
    x, y = start_point
    while -bound <= x < width + bound and -bound <= y < height + bound: # 画像の範囲内で
        if 0 <= x < width and 0 <= y < height:
            attenuation += image[int(y), int(x)] # 現時点のサンプル点の画素値を加算
            if debug:
                history.append([int(x), int(y)])
        x += cos_angle # サンプル点を1進める(xはcos_angle分だけ進む)
        y += sin_angle # (yはsin_angle分だけ進む)
    
    # レイキャスティングの履歴を表示
    if debug:
        histimg = np.zeros_like(image)
        for x, y in history:
            histimg[y, x] = 1
        plt.imshow(histimg, cmap='gray')
        plt.show()


    return attenuation

def rotate(point, origin, angle):
    """
    originを中心に、pointをangle分回転させる。
    
    :param point: 回転させる点
    :param origin: 回転の中心
    :param angle: 回転角度（ラジアン）
    :return: 回転後の点
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



if __name__ == '__main__':

    # 画像の読み込み(&表示)
    ref = np.loadtxt('reference.txt')
    plt.imshow(ref, cmap=cm.Greys_r)

    # 各種パラメータの設定
    img_size = ref.shape[0] # 画像サイズ
    n_proj = 180 # 投影数
    n_det = 200 # 検出器の数
    sod, sdd = 300, 600 #　sod:線源とオブジェクト中心の距離, sdd: 線源と検出器の距離

    theta_list = np.linspace(0, 2 * np.pi, n_proj) # 投影角度のリスト
    phi_list = np.arctan(np.linspace(-n_det / 2, n_det / 2, n_det) / sdd) # 検出器の各ピクセルの角度

    print(theta_list)
    print(phi_list)

    # サイノグラムの作成
    sinogram = np.zeros((n_proj, n_det)) # サイノグラムは、投影数x検出器数の画像

    # ray-castingの要領で、各投影を計算
    tmp_x = img_size / 2 - sod
    tmp_y = img_size / 2
    start_point_orig = np.array([tmp_x, tmp_y])

    for i in range(n_proj):
        start_point = rotate(start_point_orig, np.array([img_size / 2, img_size / 2]), theta_list[i])
        
        for j in range(n_det):
            attenuation = simple_ray_casting(ref, start_point, theta_list[i] + phi_list[j], sod)
            #attenuation = simple_ray_casting(ref, start_point_orig, phi_list[j], sod, debug=True)
            sinogram[i, j] = attenuation

    # サイノグラムの表示
    plt.figure()
    plt.imshow(sinogram, cmap=cm.Greys_r)
    plt.show()

    # サイノグラムの保存
    np.savetxt('sinogram.txt', sinogram)





