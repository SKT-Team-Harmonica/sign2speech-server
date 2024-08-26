import torch
import torch.nn as nn

# Normalization Layer Getter
def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN_1d": nn.BatchNorm1d,
            "BN_2d": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]

# Conv1d Wrapper
class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# Conv2d Wrapper
class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# 모델 클래스 정의
class SignModel(nn.Module):
    def __init__(self, vocab):
        super(SignModel, self).__init__()
        activation = nn.ReLU()

        # 2D Feature Extraction
        self.conv1 = Conv2d(3, 32, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(32))
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = Conv2d(32, 64, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(64))
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = Conv2d(64, 64, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(64))
        self.conv4 = Conv2d(64, 128, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(128))
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5 = Conv2d(128, 128, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(128))
        self.conv6 = Conv2d(128, 256, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(256))
        self.pool6 = nn.MaxPool2d(2, stride=2)

        self.conv7 = Conv2d(256, 256, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(256))
        self.conv8 = Conv2d(256, 512, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(512))
        self.pool8 = nn.MaxPool2d(2, stride=2)

        self.conv9 = Conv2d(512, 512, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(512))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Temporal Encoding
        self.tconv1 = Conv1d(512, 512, 5, stride=1, padding=2, activation=activation, norm=nn.BatchNorm1d(512))
        self.tpool1 = nn.MaxPool1d(2, stride=2)

        self.tconv2 = Conv1d(512, 512, 5, stride=1, padding=2, activation=activation, norm=nn.BatchNorm1d(512))
        self.tpool2 = nn.MaxPool1d(2, stride=2)

        self.tconv3 = Conv1d(512, 1024, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm1d(1024))

        # Classification
        self.classifier = nn.Linear(1024, len(vocab))

        # Initialize layers
        self.init_layers()

    def init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 2D Feature Extraction
        x = self.extract_feature(x)

        # Temporal Encoding
        x = self.tpool1(self.tconv1(x))
        x = self.tpool2(self.tconv2(x))
        x = self.tconv3(x)  # (batch, 1024, T//4)

        # Classification
        x = x.transpose(1, 2)  # (batch, T//4, 1024)
        x = self.classifier(x)  # (batch, T//4, C)
        return x

    def extract_feature(self, x):
        batch_size, clip_length, C, H, W = x.shape
        x = x.view(batch_size * clip_length, C, H, W)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool4(self.conv4(self.conv3(x)))
        x = self.pool6(self.conv6(self.conv5(x)))
        x = self.pool8(self.conv8(self.conv7(x)))
        x = self.avg_pool(self.conv9(x))  # (B*T, 512, 1, 1)
        x = x.view(x.shape[:2]).view(batch_size, clip_length, -1).transpose(1, 2)  # (B, C, T)
        return x

# 모델 로드 함수
def load_model(model_path: str, model_class: nn.Module, vocab: list) -> nn.Module:
    try:
        model = model_class(vocab)  # 모델 클래스의 인스턴스를 생성합니다.

        # 전체 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # 모델 가중치만 로드, classifier 레이어는 무시
        state_dict = checkpoint['state_dict']
        # classifier 레이어의 가중치 및 바이어스를 제거
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}

        # 나머지 가중치를 모델에 로드
        model.load_state_dict(state_dict, strict=False)

        # classifier 레이어의 크기 불일치로 인한 오류 해결: 현재 vocab 크기에 맞게 다시 초기화
        model.classifier = nn.Linear(1024, len(vocab))

        model.eval()  # 모델을 평가 모드로 전환
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
