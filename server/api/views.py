import cv2
import numpy as np
import torch
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from .pytorch import model


class ProcessImageView(APIView):
    device = torch.device('cuda')

    def post(self, request: Request):

        file = request.FILES.get('img')

        try:
            img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32)
            img = img / 255.0

            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            img = img.transpose((2, 0, 1))

            t_img = torch.from_numpy(img).to(self.device)
            t_img = t_img.unsqueeze(0)

            with torch.no_grad():
                output = model(t_img).squeeze(0)

                if output[0] > 0:
                    return Response("Most likely provided picture contains dog")
                else:
                    return Response("Most likely provided picture contains cat")

        except Exception as e:
            raise Exception(f"Something went wrong, {e}")
# Create your views here.
