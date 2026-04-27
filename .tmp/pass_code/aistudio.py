# -*- encoding: utf-8 -*-
import os
import logging
import numpy as np
import os
import torch
from pathlib import Path
import base64
import sys
import cv2
import json
import ast


import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy


from gr00t.experiment.data_config import DATA_CONFIG_MAP

# print(DATA_CONFIG_MAP.keys())

# 统一继承MayaBaseHandler，可以自动获取组件面板的模型文件
from aistudio_serving.hanlder.pymps_handler import MayaBaseHandler

# 组件使用文档，详见 https://yuque.antfin-inc.com/aii/aistudio/nkyse5
# python lib 依赖写入 requirement.txt

logger = logging.getLogger()



# 用户自定义代码处理类
class UserHandler(MayaBaseHandler):
    """
     model_dir: model.py 文件所在目录
    """

    def __init__(self, model_dir):
        # 父类初始化
        super(UserHandler, self).__init__(model_dir)

        # 可以认为 self.resource_path 就是上游组件输入的模型或者python组件面板设置的 '自定义资源地址'在本地磁盘的路径，如果都没有返回 None
        # model_path = os.path.join(self.resource_path, "xxx")
        MODEL_PATH = self.get_base_path()
        print("model_path: ", MODEL_PATH)
        # print("model_path: ", self.resource_path)
        EMBODIMENT_TAG = "new_embodiment"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("device: ", device)

        data_config = DATA_CONFIG_MAP["new_interaction_group"]

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        model_left = os.path.join(MODEL_PATH, "left_hand_v2_1223/")
        self.policy_left = Gr00tPolicy(
            model_path=model_left,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=modality_config,
            modality_transform=modality_transform,
            device=device,
        )

    """
     测试demo
     1 输入配置
        query:TYPE_STRING:[1]        对应bytes 类型
     2 输出配置
        out_float:TYPE_FP64:[1]        对应float64 类型
        out_string:TYPE_STRING:[1]     对应bytes_ 类型
        out_int:TYPE_INT32:[1].        对应int32 类型
     其他
     1. TYPE_STRING对应是python bytes类型(或者np.bytes_)，目标是方便传递二进制内容，比如图片的binary内容，减少base64转换开销;
        bytes类型可以通过decode函数明确转换成python str类型
     2. 参数维度见使用文档
    """

    def predict_np(self, features, trace_id):

        logger.info("[vla]starting to predict np.")
        restored_request = json.loads(features.get("query").decode())
        # logger.info("[vla]restored_request: %s", restored_request)

        joint_angles = restored_request.get("joint_angles", b'[]')
        predicted_coords_2d = restored_request.get("predicted_coords_2d", b'[]')

        history_angles = restored_request.get("history_angles", b'[]')

        device_id = restored_request.get("device_id", b'')
        request_id = restored_request.get("request_id", b'')
        

        if (type(joint_angles) is str):
            joint_angles = ast.literal_eval(joint_angles)

        if (type(predicted_coords_2d) is str):
            predicted_coords_2d = ast.literal_eval(predicted_coords_2d)

        # 解析framebuffer
        framebuffer_data = None


        framebuffer_raw = restored_request.get("framebuffer", [])
        logger.info("[vla]--framebuffer_raw type: %s, length: %s", type(framebuffer_raw), len(framebuffer_raw) if framebuffer_raw else 0)
        
        framebuffer_size = restored_request.get("framebuffer_size", 0)  # 添加这行

        # framebuffer_encoding = restored_request.get("framebuffer_encoding", "")
        # framebuffer_size = restored_request.get("framebuffer_size", 0)
        logger.info("[vla]--starting to decode image buffer")

        # jpeg_array = np.frombuffer(framebuffer_raw, dtype=np.uint8)
        jpeg_array = np.array(framebuffer_raw, dtype=np.uint8)
        framebuffer_data = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)

        logger.info("Successfully decoded framebuffer: size=%d, expected=%d", len(framebuffer_data), framebuffer_size)
        # frame_rgb = framebuffer_data.reshape(480, 640, 3)
        frame_bgr = cv2.resize(framebuffer_data, (480, 640))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #cv2.COLOR_BGR2RGB
        # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./output.jpg', frame_bgr)

        

        # if (len(joint_angles) != 5 or len(request_id) != 16):
        #     resultMap = {
        #         "action_sequence": "",  # 执行序列
        #         "joint_angles": "",
        #         "predicted_coords_2d": "",
        #         "history_angles": history_angles,
        #         "is_success": False,
        #         "error": "joint_angles or predicted_coords_2d or request_id data error",
        #         "key_infos": "key_infos",  # json序列化字符串, key_infos数据示例如下，如不需要使用可以不传或者置空字符串
        #         "device_id": device_id,  # device_id
        #         "request_id": request_id  # request_id
        #     }
        #     return (1, "fail", resultMap)

        if (type(history_angles) is str):
            history_angles = ast.literal_eval(history_angles)
        
        # frame_rgb = np.full((480, 640, 3), 0, dtype=np.uint8)
        logger.info(f"[vla]framebuffer_size: before {joint_angles}, len: {len(joint_angles)}, joint angles type: {type(joint_angles)}")
        res_list = []
        for ids in range(5):
            res_list.append(joint_angles[ids])
        res_list.append(0)
        robot_state = np.array(res_list, dtype=np.float32)

        # frame_rgb2 = np.full((480, 640, 3), 255, dtype=np.uint8)
        frame_send =  frame_rgb[np.newaxis, ...]
        logger.info(f"[vla]framebuffer_size: after,{type(robot_state)}, {robot_state}, {frame_send.shape},{type(frame_send)}")
        # "Keep the book centered in the camera view."
        # 将BGR图片转换为灰度图
        batch = {
            "video.ego_view": frame_send,  # 添加batch维度
            "annotation.task_index": ["Move to center the book in view. Do nothing if no book is present."],
            "state.single_arm": robot_state[np.newaxis, ...],  # 添加batch维度
        }
        
        previous_actions = self.policy_left.get_action(batch)['action.single_arm']
        
        # previous_actions = self.policy.get_action(batch)['action.single_arm']
        # print("previous_actions: ", previous_actions)

        pred_angles_5dof = previous_actions[:, :6]
        previous_actions = np.cumsum(pred_angles_5dof, axis=0) + robot_state[:6]

        # 先转换为Python list，再逐个round
        rounded_actions = [[round(float(x), 4) for x in row] for row in previous_actions[:14]] # 9 is right
        action_string = json.dumps(rounded_actions)

        # print("previous_actions: ", previous_actions)


        # print("action_string: ", action_string)
        joint_angles = json.dumps(joint_angles)
        predicted_coords_2d = json.dumps(predicted_coords_2d)
        history_angles = json.dumps(history_angles)
        if (previous_actions.size < 5):
            is_success = False
        else:
            is_success = True

        resultMap = {
            "action_sequence": action_string,  # 执行序列
            "joint_angles": joint_angles,
            "predicted_coords_2d": predicted_coords_2d,
            "history_angles": history_angles,
            "is_success": is_success,
            "error": "",
            "key_infos": "key_infos",  # json序列化字符串, key_infos数据示例如下，如不需要使用可以不传或者置空字符串
            "device_id": device_id,  # device_id
            "request_id": request_id  # request_id
        }

        logger.info("predict result: %s", json.dumps(resultMap))  # 可以在平台查看相关日志

        # 处理结果返回
        resultCode = 0  # 0表示成功，其它为失败
        errorMessage = "ok"  # errorMessage为predict函数对外透出的信息
        return (resultCode, errorMessage, resultMap)


# 用于调试UserHandler类的功能
if __name__ == "__main__":
    # 示例
    import os

    # 使用绝对路径初始化
    user_handler = UserHandler(os.getcwd())
    # str 类型使用 .encode() 编码模拟引擎调用的真实输入, 代码中进行decode()
    # {
    # "joint_angles": list,  # 关节角
    # "predicted_coords_2d": list,  # 预测手眼2d坐标
    # "image_id": str,  # 请求对应图片的唯一标
    # "diagnostic_id": str  # 诊断排查id
    # }

    request = {}

    joint_angles = [10, 10, 10, 10, 10]
    # joint_angles = []
    history_angles = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]]
    history_angles = []
    predicted_coords_2d = [0.5, 0.5]
    # predicted_coords_2d = []
    label_coord = ""
    device_id = ""
    request_id = "1234567890123456"

    request["joint_angles"] = str(joint_angles)
    request["predicted_coords_2d"] = str(predicted_coords_2d)
    request["label_coord"] = str(label_coord)
    # request["diagnostic_id"] = str(diagnostic_id)
    request["history_angles"] = str(history_angles)
    request["device_id"] = str(device_id)
    request["request_id"] = str(request_id)


    request["framebuffer"] = "aaa"
    vla = {}
    vla["query"] = json.dumps(request).encode()  # 转成 JSON 字符串
    print(vla)

    user_handler.predict_np(vla, "test_trace_id")
