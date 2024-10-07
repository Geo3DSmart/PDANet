import open3d
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

class mycolormap:
    # colormaps = ['viridis','Reds', 'Blues', 'Greens', 'Purples', 'Oranges'] # , 'Greys'
    colormaps = ['brg','viridis','Blues', 'Greens','Reds',  'Purples']
    colormap_ins = [0] * 6
    def __init__(self) -> None:
        for i, colormap in enumerate(self.colormaps):
            self.colormap_ins[i] = self.get_colormap(colormap)
    def get_colormap(self, name):	
        return plt.get_cmap(name, 11)([i for i in range(11)])[:, 0:3]


mycolormap_ins = mycolormap()



def key_forward_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(1, 0, 0)
    return True

def key_back_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(-1, 0, 0)
    return True

def key_left_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, -1, 0)
    return True

def key_right_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 1, 0)
    return True

def key_up_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, 1)
    return True

def key_down_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, -1)
    return True

def key_look_up_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(0,-10)
    return True

def key_look_down_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(0,10)
    return True

def key_look_right_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(20,0)
    return True

def key_look_left_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(-20,0)
    # ctr.set_up([0,0,1])
    return True

def key_reset_up_callback(vis):
    ctr = vis.get_view_control()
    ctr.set_up([0,0,1])
    ctr.reset_camera_local_rotate()
    return True

def get_color(segscores):
    global mycolormap_ins

    size = segscores.shape[0]
    colors = np.zeros((size, 3))
    for i, segscore in enumerate(segscores):
        max_score = np.max(segscore)
        # if max_score>0.2:
        #     colorlevel = int(max_score / 0.1)+10
        # else:
        colorlevel = int(max_score / 0.1) 
        index = np.where(segscore==max_score)
        index = index[0][0]
        colormap = mycolormap_ins.colormap_ins[index]
        colors[i] = colormap[colorlevel]
    return colors

def read_display_pcd_pc_single(pcd, score, vis):
    pcd.paint_uniform_color([1, 1, 1])
    single = np.array([np.array([0, 0, 1])   for _ in range(get_color(score).shape[0])])
    pcd.colors = open3d.utility.Vector3dVector(single )

    vis.add_geometry(pcd)

def read_display_pcd_pc(pcd, score, vis):
    pcd.paint_uniform_color([1, 1, 1])
    
    pcd.colors = open3d.utility.Vector3dVector(get_color(score))

    vis.add_geometry(pcd)

def get_vis_single(pcd_data,score):
    # pcd 转为 numpy
    # score 转为numpy
    pcd_data =  pcd_data.cpu().numpy()
    pcd =open3d.geometry.PointCloud()
    # pcd=open3d.io.read_point_cloud(pcd_path) 
    pcd.points =open3d.utility.Vector3dVector(pcd_data)
    # score = score.cpu().numpy()
    score = score.cpu().numpy()
    
    max_score = np.max(score)
    score = score / max_score
    
    vis=open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="point_cloud", width=1920, height=1080)
    
    opt=vis.get_render_option()  
    # opt.background_color = np.array([0, 0, 0])
    # opt.background_color = np.array([255, 255, 255])
    opt.background_color = np.array([200, 200, 200])
    opt.point_size = 3.0
    # opt.load_from_json(os.path.dirname(__file__) + './RenderOption.json')

    vis.reset_view_point(True)

    read_display_pcd_pc_single(pcd, score, vis)
    


    vis.poll_events()
    vis.update_renderer()

    vis.register_key_callback(ord('W'), key_forward_callback)
    vis.register_key_callback(ord('S'), key_back_callback)
    vis.register_key_callback(ord('A'), key_left_callback)
    vis.register_key_callback(ord('D'), key_right_callback)
    vis.register_key_callback(ord(' '), key_up_callback)
    vis.register_key_callback(ord('C'), key_down_callback)
    vis.register_key_callback(ord('I'), key_look_up_callback)
    vis.register_key_callback(ord('K'), key_look_down_callback)
    vis.register_key_callback(ord('J'), key_look_left_callback)
    vis.register_key_callback(ord('L'), key_look_right_callback)
    vis.register_key_callback(ord('F'), key_reset_up_callback)


    vis.run()


#     # pcd点云的分数为 [N,3]
#     # 点云pcd文件   [N,3]
def get_vis(pcd_data,score):
    # pcd 转为 numpy
    # score 转为numpy
    pcd_data =  pcd_data.cpu().numpy()
    pcd =open3d.geometry.PointCloud()
    # pcd=open3d.io.read_point_cloud(pcd_path) 
    pcd.points =open3d.utility.Vector3dVector(pcd_data)
    # score = score.cpu().numpy()
    score = score.cpu().numpy()
    
    max_score = np.max(score)
    score = score / max_score
    
    vis=open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="point_cloud", width=1920, height=1080)
    
    opt=vis.get_render_option()  
    # opt.background_color = np.array([0, 0, 0])
    # opt.background_color = np.array([255, 255, 255])
    opt.background_color = np.array([200, 200, 200])
    opt.point_size = 3.0
    # opt.load_from_json(os.path.dirname(__file__) + './RenderOption.json')

    vis.reset_view_point(True)

    read_display_pcd_pc(pcd, score, vis)
    


    vis.poll_events()
    vis.update_renderer()

    vis.register_key_callback(ord('W'), key_forward_callback)
    vis.register_key_callback(ord('S'), key_back_callback)
    vis.register_key_callback(ord('A'), key_left_callback)
    vis.register_key_callback(ord('D'), key_right_callback)
    vis.register_key_callback(ord(' '), key_up_callback)
    vis.register_key_callback(ord('C'), key_down_callback)
    vis.register_key_callback(ord('I'), key_look_up_callback)
    vis.register_key_callback(ord('K'), key_look_down_callback)
    vis.register_key_callback(ord('J'), key_look_left_callback)
    vis.register_key_callback(ord('L'), key_look_right_callback)
    vis.register_key_callback(ord('F'), key_reset_up_callback)


    vis.run()
    
    


# if __name__=="__main__":
#     # todo 点云输入方式，从命令行输入
#     parser = argparse.ArgumentParser(description='pc visualize')
#     parser.add_argument('pcd', type=str, default='test.pcd')
#     args = parser.parse_args()
#     pcd_path = args.pcd    
#     pcd=open3d.io.read_point_cloud(pcd_path) 
#     # * 随机生成分数
#     size = np.asarray(pcd.points).shape[0]
#     score = np.random.rand(size, 3)

#     # todo 点云输入方式，从代码输入，np数组：pcd[N, 3]， score[N, num_class]
#     # pcd = open3d.utility.Vector3dVector(pcd)
#     # pcd = open3d.cpu.pybind.geometry.PointCloud(pcd)
#     # score = score
    

#     # pcd点云的分数为 [N,3]
#     # 点云pcd文件   [N,3]



#     vis=open3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(window_name="point_cloud", width=1920, height=1080)
    
#     opt=vis.get_render_option()  
#     opt.background_color = np.array([0, 0, 0])
#     opt.point_size = 2.0
#     # opt.load_from_json(os.path.dirname(__file__) + './RenderOption.json')

#     vis.reset_view_point(True)

#     read_display_pcd_pc(pcd, score, vis)
    


#     vis.poll_events()
#     vis.update_renderer()

#     vis.register_key_callback(ord('W'), key_forward_callback)
#     vis.register_key_callback(ord('S'), key_back_callback)
#     vis.register_key_callback(ord('A'), key_left_callback)
#     vis.register_key_callback(ord('D'), key_right_callback)
#     vis.register_key_callback(ord(' '), key_up_callback)
#     vis.register_key_callback(ord('C'), key_down_callback)
#     vis.register_key_callback(ord('I'), key_look_up_callback)
#     vis.register_key_callback(ord('K'), key_look_down_callback)
#     vis.register_key_callback(ord('J'), key_look_left_callback)
#     vis.register_key_callback(ord('L'), key_look_right_callback)
#     vis.register_key_callback(ord('F'), key_reset_up_callback)


#     vis.run()