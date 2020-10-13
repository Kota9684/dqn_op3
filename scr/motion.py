#!/usr/bin/env python
# coding: utf-8
from op3_controller.msg import Command

class Motion:
    def __init__(self,array):
        self.array = array

    def motion(self,num):
        #初期状態
        # if num == 17:
        #     #肘
        #     self.array.l_el = 0.0 #-1.5
        #     self.array.r_el = 0.0 #1.5
        #     #左肩
        #     self.array.l_sho_pitch = 0.00 # -1.7
        #     self.array.l_sho_roll =  0.00
        #     #右肩
        #     self.array.r_sho_pitch = 0.00 #1.7
        #     self.array.r_sho_roll =  0.00
        #     #左腰
        #     self.array.l_hip_pitch = 0.00 #-0.6
        #     self.array.l_hip_roll = 0.00
        #     self.array.l_hip_yaw = 0.00
        #     #右腰
        #     self.array.r_hip_pitch = 0.00 #0.6
        #     self.array.r_hip_roll = 0.00
        #     self.array.r_hip_yaw = 0.00
        #     #左ひざ
        #     self.array.l_knee = 0.00
        #     #右ひざ
        #     self.array.r_knee = 0.00
        #     #左足首
        #     self.array.l_ank_pitch = 0.00
        #     self.array.l_ank_roll = 0.00
        #     #右足首
        #     self.array.r_ank_pitch = 0.00
        #     self.array.r_ank_roll  = 0.00
        ###########################################################

        if num == 0:
            #左膝プラス
            self.array[12] += 0.3
            if self.array[12] >0.3:
                self.array[12] =0.3

        elif num == 1:
            #左膝マイナス
            self.array[12] -= 0.3
            if self.array[12] <-0.3:
                self.array[12] =-0.3

        elif num == 2:
            #右膝プラス
            self.array[13] += 0.3
            if self.array[13] >0.3:
                self.array[13] =0.3

        elif num == 3:
            #右膝マイナス
            self.array[13] -= 0.3
            if self.array[13] <-0.3:
                self.array[13] =-0.3

        elif num == 4:
            #左腰プラス
            self.array[6] += 0.3
            if self.array[6] > 0.3:
                self.array[6] = 0.3

        elif num == 5:
            #左腰マイナス
            self.array[6] -= 0.3
            if self.array[6] < -0.3:
                self.array[6] = -0.3

        elif num == 6:
            #右腰プラス
            self.array[9] += 0.3
            if self.array[9] > 0.3:
                self.array[9] = 0.3

        elif num == 7:
            #右腰マイナス
            self.array[9] -= 0.3
            if self.array[9] < -0.3:
                self.array[9] = -0.3

        elif num == 8:
            #左足首プラス
            self.array[14] += 0.06
            if self.array[14] >1/2:
                self.array[14] =1/2

        elif num == 9:
            #左足首マイナス
            self.array[14] -= 0.06
            if self.array[14] <-0.5:
                self.array[14] =-0.5

        elif num == 10:
            #右足首プラス
            self.array[16] += 0.06
            if self.array[16] >0.5:
                self.array[16] =0.5

        elif num == 11:
            #右足首マイナス
            self.array[16] -= 0.06
            if self.array[16] <-0.5:
                self.array[16] =-0.5

        # elif num == 12:
        #     #左肩プラス
        #     self.array[2] += 0.1
        #     if self.array[3] > 1/2:
        #         self.array[3] = 1/2
        #
        # elif num == 13:
        #     #左肩マイナス
        #     self.array[2] -= 0.1
        #     if self.array[3] < -1/2:
        #         self.array[3] = -1/2
        #
        # elif num == 14:
        #     #右肩プラス
        #     self.array[4] += 0.1
        #     if self.array[5] > 1/2:
        #         self.array[5] = 1/2
        #
        # elif num == 15:
        #      #右肩マイナス
        #     self.array[4] -= 0.1
        #     if self.array[5] < -1/2:
        #         self.array[5] = -1/2


        # elif num == 12:
        #     #左肘プラス
        #     self.array[0] += 0.1
        #     if self.array[0] > 1/2:
        #         self.array[0] = 1/2
        #
        # elif num == 13:
        #     #左肘マイナス
        #     self.array[0] -= 0.1
        #     if self.array[0] < -1/2:
        #         self.array[0] = -1/2
        #
        # elif num == 14:
        #     #右肘プラス
        #     self.array[1] += 0.1
        #     if self.array[1] > 1/2:
        #         self.array[1] = 1/2
        #
        # elif num == 15:
        #     #右肘マイナス
        #     self.array[1] -= 0.1
        #     if self.array[1] < -1/2:
        #         self.array[1] = -1/2

        return self.array
