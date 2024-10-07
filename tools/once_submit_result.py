
import pickle




filename_submit = r"once_sumit_sample/result.pkl"


# pred_result
filename = r"/mnt/data/tm/code/3ddetection/S_semant_model/IA-SSD_PVRCNN_Centerpoint/output/once_models/pvrcnn_IASSDAddmodel/default/eval/epoch_76/test/default/result.pkl"



F=open(filename_submit,'rb')
content=pickle.load(F)
filename_submit_frame_id = []
for index_name in content:
    frame_id = index_name['frame_id']
    filename_submit_frame_id.append(frame_id)
    # print("")

store_final = []
F=open(filename,'rb')
content=pickle.load(F)
for index_name in content:
    all_dict = {}
    frame_id = index_name['frame_id']
    
    mask_score  = index_name['score']>0.2
    
    name = index_name['name'][mask_score]
    score = index_name['score'][mask_score]
    boxes_3d = index_name['boxes_3d'][mask_score]
    
    all_dict['name'] = name
    all_dict['score'] = score
    all_dict['boxes_3d'] = boxes_3d
    all_dict['frame_id'] = frame_id
    
    
    if frame_id in filename_submit_frame_id:
        store_final.append(all_dict)
        # store_final.append(index_name)
 
db_info_save_path = './result.pkl'
with open(db_info_save_path, 'wb') as f:
    pickle.dump(store_final, f)


# 7897
db_info_save_path = './result.pkl'
F=open(db_info_save_path,'rb')
content=pickle.load(F)
print(len(content))

print("asdasd")