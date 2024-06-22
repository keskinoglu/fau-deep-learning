import numpy as np
from skimage import measure
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
    
    def forward(self, input_tensor):
        # stride_y = int(np.ceil(input_tensor.shape[2] / self.stride_shape[0]))
        # stride_x = int(np.ceil(input_tensor.shape[3] / self.stride_shape[1]))
        # output_tensor = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
        #                         stride_y, stride_x))
        
        # temp_y = (input_tensor.shape[2] - self.stride_shape[0])
        # temp_x = (input_tensor.shape[3] - self.stride_shape[1])
        # temp_op = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
        #                     temp_y, temp_x))

        # for b in range(input_tensor.shape[0]):
        #     for c in range(input_tensor.shape[1]):
        #         print((measure.block_reduce(input_tensor[b, c], self.pooling_shape[0], np.max)).shape)
        #         temp_op[b, c] = measure.block_reduce(input_tensor[b, c], self.pooling_shape[0], np.max)
        
        # #subsampling
        # for b in range(input_tensor.shape[0]):
        #     for c in range(input_tensor.shape[1]):
        #         output_tensor[b, c] = temp_op[b, c][::self.stride_shape[0], ::self.stride_shape[1]]

        # return(output_tensor)
        self.pool_input_shape = input_tensor.shape
        op_y = int(np.floor(((input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0]) + 1))
        op_x = int(np.floor(((input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1]) + 1))
        op_shape = (input_tensor.shape[0], input_tensor.shape[1], op_y, op_x)
        output_tensor = np.zeros((op_shape))
        self.index_dict = {}
        self.max_idx = []

        # for b in range(input_tensor.shape[0]):
        #     for c in range(input_tensor.shape[1]):
        #         max_values = []
        #         pools = []
        #         for i in np.arange(input_tensor.shape[2], step=self.stride_shape[0]):
        #             for j in np.arange(input_tensor.shape[3], step=self.stride_shape[1]):
        #                 # Extract the current pool
        #                 cur_pool = input_tensor[b, c][i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]]
                        
        #                 if cur_pool.shape == (self.pooling_shape[0], self.pooling_shape[1]):
        #                     pools.append(cur_pool)
        #         for pool in pools:
        #             max_value = np.max(pool)
        #             # self.max_idx.append((b, c) + np.unravel_index(np.argmax(pool, axis=None), pool.shape))
        #             y, x = np.where(input_tensor[b, c] == max_value)
        #             flag = False
        #             for i in y:
        #                 for j in x:
        #                     if (b, c) + (i, j) not in self.max_idx:
        #                         self.max_idx.append((b, c) + (i, j))
        #                         flag = True
        #             # if flag == False:
        #             #     self.max_idx.append((b, c) + (y[-1], x[-1]))

        #             self.max_idx.append((b, c) + (y[0], x[0]))
        #             # self.index_dict[b, c, i[0], j[0]] = max_value
        #             max_values.append(max_value)
        #             # max_values.append(np.max(pool))

        #         output_tensor[b, c] = np.array(max_values).reshape((op_y, op_x))
                
        # return(output_tensor)


        for b in range(input_tensor.shape[0]):
            for c in range(input_tensor.shape[1]):
                max_values = []
                # pools = []
                for i in np.arange(input_tensor.shape[2], step=self.stride_shape[0]):
                    for j in np.arange(input_tensor.shape[3], step=self.stride_shape[1]):
                        # Extract the current pool
                        cur_pool = input_tensor[b, c][i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]]
                        
                        if cur_pool.shape == (self.pooling_shape[0], self.pooling_shape[1]):
                            # pools.append(cur_pool)
                            max_value = np.max(cur_pool)
                            y, x = np.where(cur_pool == max_value)
                            y_idx = y[0] + i
                            x_idx = x[0] + j
                            self.max_idx.append((b, c) + (y_idx, x_idx))
                            # max_value = np.max(cur_pool)
                            max_values.append(max_value)

                output_tensor[b, c] = np.array(max_values).reshape((op_y, op_x))
                
        return(output_tensor)


    def backward(self, error_tensor):
        # pass
        et = error_tensor.flatten()
        print(et.shape, self.max_idx)
        output_tensor = np.zeros((self.pool_input_shape))
        for key, max_ele in zip(self.max_idx, et):
            # output_tensor[b, c, y, x] = self.index_dict[b, c, y, x]
            output_tensor[key] = output_tensor[key] + max_ele
        # print(output_tensor.shape)
        return output_tensor


        # et = error_tensor.flatten()
        # output_tensor = np.zeros((self.pool_input_shape))
        # for key, max_ele in zip(self.index_dict, et):
        #     # output_tensor[b, c, y, x] = self.index_dict[b, c, y, x]
        #     output_tensor[key[0], key[1], key[2], key[3]] = max_ele
        # print(output_tensor.shape)
        # return output_tensor

        # output_tensor = np.zeros((self.pool_input_shape))
        # max_values = []
        # for b in range(error_tensor.shape[0]):
        #     for c in range(error_tensor.shape[1]):
        #         for y in range(error_tensor.shape[2]):
        #             for x in range(error_tensor.shape[3]):
        #                 max_values.append(error_tensor[b,c,y,x])
        
        # for key, val in zip(self.index_dict, max_values):
        #     output_tensor[key[0], key[1], key[2], key[3]] = val
        # return output_tensor

            


