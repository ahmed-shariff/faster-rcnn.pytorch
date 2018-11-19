from __future__ import absolute_import
import torch
import numpy as np
try:
    from ._ext import nms
    def nms_gpu(dets, thresh):
        keep = dets.new(dets.size(0), 1).zero_().int()
        num_out = dets.new(1).zero_().int()
        nms.nms_cuda(keep, dets, num_out, thresh)
        keep = keep[:num_out[0]]
        return keep

except ImportError:
    # nms implemented using basic torch functions.
    def nms_gpu(dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        _, order = scores.sort(descending = True)

        keep = []
        while order.size(0) > 0:
            idx = order[0]
            
            #using .item() here to avoid copies of tensors being made.
            keep.append(idx.item())
            intersection_x1 = torch.max(x1[idx], x1[order[1:]])
            intersection_x2 = torch.min(x2[idx], x2[order[1:]])
            intersection_y1 = torch.max(y1[idx], y1[order[1:]])
            intersection_y2 = torch.min(y2[idx], y2[order[1:]])

            w = torch.clamp(intersection_x2 - intersection_x1 + 1, min = 0)
            h = torch.clamp(intersection_y2 - intersection_y1 + 1, min = 0)
            intersection = h * w
            union = areas[idx] + areas [order[1:]] - intersection
            iou = intersection / union

            remaining_indices = iou < thresh
            order = order[1:][remaining_indices]

        return torch.IntTensor(keep)
        
