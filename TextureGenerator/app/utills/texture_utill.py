import trimesh
from collections import Counter

import numpy as np
from texture_baker import TextureBaker

import torch
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt


def extract_mesh(glb_file_path):
    # GLB 파일 로드 (Scene 객체)
    scene = trimesh.load(glb_file_path)

    # Scene에서 모든 메쉬 추출 (여러 메쉬가 있을 수 있음)
    meshes = scene.dump()

    # 첫 번째 메쉬만 사용 (필요에 따라 반복문으로 여러 메쉬 처리 가능)
    mesh = meshes[0]

    # 메쉬의 정점 좌표, UV 좌표, 삼각형 인덱스 추출
    mesh_v_pos = mesh.vertices  # 정점 좌표 (vertices)
    mesh_v_tex = mesh.visual.uv  # UV 좌표 (UV coordinates)
    mesh_t_pos_idx = mesh.faces  # 삼각형 인덱스 (faces)

    # 추출된 데이터를 딕셔너리 형식으로 반환
    mesh_data = {
        "v_pos": mesh_v_pos,
        "v_tex": mesh_v_tex,
        "t_pos_idx": mesh_t_pos_idx
    }
    
    return mesh_data

# UV좌표 중복 체크 함수
def check_duplicate_uv(mesh_data):
    uv_coords = mesh_data['v_tex']

    # UV 좌표를 튜플로 변환 (numpy array는 hashable하지 않기 때문에 튜플로 변환)
    uv_tuples = [tuple(uv) for uv in uv_coords]

    # UV 좌표의 중복 여부 확인
    uv_counter = Counter(uv_tuples)

    # 중복된 UV 좌표를 찾음
    duplicates = [uv for uv, count in uv_counter.items() if count > 1]

    if duplicates:
        print(f"중복된 UV 좌표가 발견되었습니다. 중복된 UV 좌표의 개수: {len(duplicates)}")
        for uv in duplicates:
            print(f"중복된 UV 좌표: {uv}, 중복 횟수: {uv_counter[uv]}")
    else:
        print("모든 UV 좌표가 고유합니다.")


def generate_texture(mesh_data, image, bake_resolution: int):  
    
    # 1. 메쉬 데이터 처리
    mesh_v_pos = torch.from_numpy(mesh_data["v_pos"]).float()
    mesh_v_tex = torch.from_numpy(mesh_data["v_tex"]).float()
    mesh_t_pos_idx = torch.from_numpy(mesh_data["t_pos_idx"]).int()
    
    # 2. 이미지 전처리
    image = image.resize((bake_resolution, bake_resolution))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    #plt.imshow(image)


    image_np = np.array(image) / 255.0
    image_tensor = torch.from_numpy(image_np).float()  # (H, W, C)
    
    # 필요 시 이미지 텐서의 차원 조정 (C, H, W) 형태로 변환)
    #image_tensor = image_tensor.permute(2, 0, 1)
    
    # 3. 텍스처 베이킹
    texture_baker = TextureBaker()
    rast = texture_baker.rasterize(mesh_v_tex, mesh_t_pos_idx, bake_resolution)
    bake_mask = texture_baker.get_mask(rast)
    pos_bake = texture_baker.interpolate(image_tensor, rast, mesh_t_pos_idx)
    
    # 마스크를 pos_bake의 마지막 차원에 맞게 확장
    bake_mask = bake_mask.unsqueeze(-1)  # (H, W) -> (H, W, 1)
    bake_mask = bake_mask.expand(-1, -1, pos_bake.shape[2])  # (H, W, 1) -> (H, W, C)
    
    # 마스크를 적용
    texture_bake = pos_bake * bake_mask  # (H, W, C) * (H, W, C)
    

    # 텐서를 NumPy 배열로 변환
    texture_bake_np = texture_bake.cpu().numpy()
    
    # 값의 범위를 0~1로 클리핑
    texture_bake_np = np.clip(texture_bake_np, 0, 1)
    
    # 0~255 값으로 변환 후 PIL 이미지로 변환 -> 기존에 만들어진 라이브러리가 있더라 적용해보자
    texture_bake_np = (texture_bake_np * 255).astype(np.uint8)

    texture_image = Image.fromarray(texture_bake_np)
    
    rast_np = rast.cpu().numpy()
    '''
    plt.imshow(rast_np)
    plt.title('Rasterization Result')
    plt.show()

    pos_bake_np = pos_bake.cpu().numpy()
    plt.imshow(pos_bake_np)
    plt.title('Interpolated Texture')
    plt.show()'''


    print(f"pos_bake shape: {pos_bake.shape}")  # 예상: (H, W, C)
    print(f"bake_mask shape: {bake_mask.shape}")  # 예상: (H, W, C)
    print(f"texture_bake shape: {texture_bake.shape}")  # 예상: (H, W, C)

    print(f"pos_bake min/max: {pos_bake.min()}, {pos_bake.max()}")
    print(f"texture_bake min/max: {texture_bake.min()}, {texture_bake.max()}")

    return texture_image


# 이미지 위에 uv좌표 표시해보기 - 그림 뭉게지는거 확인용
def visualize_uv_on_texture(image_path, uv_coords):
    # 이미지 불러오기
    image = Image.open(image_path)
    image = np.array(image)

    # 플롯 설정
    plt.figure(figsize=(10, 10))
    plt.imshow(image)  # 이미지를 배경으로 표시

    # UV 좌표를 [0, 1] 범위에서 이미지 픽셀 크기에 맞게 변환
    img_height, img_width, _ = image.shape
    uv_coords_scaled = uv_coords.copy()
    uv_coords_scaled[:, 0] *= img_width   # U 좌표 (x축)
    uv_coords_scaled[:, 1] = 1 - uv_coords_scaled[:, 1]  # V 좌표는 위에서 아래로 가는 방향이므로 1에서 빼줌
    uv_coords_scaled[:, 1] *= img_height  # V 좌표 (y축)

    # UV 좌표를 시각적으로 표시 (점으로 표시)
    plt.scatter(uv_coords_scaled[:, 0], uv_coords_scaled[:, 1], c='r', s=10, label="UV Coordinates")
    
    plt.title("UV Mapping on Texture Image")
    plt.legend()
    plt.axis('off')  # 축을 제거하여 시각적으로 더 보기 좋게
    plt.show()

def create_texture_mask(mesh_data, texture_size=(512, 512)):
    # 메쉬의 UV 좌표 추출
    uv_coords = mesh_data["v_tex"]
    
    # 텍스처 마스크 초기화 (0으로 초기화된 이미지)
    mask = Image.new("L", texture_size, 0)
    draw = ImageDraw.Draw(mask)

    # UV 좌표를 텍스처 마스크 크기에 맞게 스케일링
    uv_coords_scaled = uv_coords * texture_size

    # UV 좌표에 따라 다각형 영역을 채움
    for face in mesh_data["t_pos_idx"]:
        # 각 face의 UV 좌표 추출 및 스케일 적용
        polygon = [(uv_coords_scaled[idx][0], uv_coords_scaled[idx][1]) for idx in face]
        draw.polygon(polygon, fill=255)  # 흰색으로 채우기

    # 텍스처 마스크 이미지 반환
    return mask

def save_uv_mask_as_image(mask_tensor, save_path="uv_mask.png"):
    """
    UV 마스크 텐서를 이미지 파일로 저장하는 함수

    Args:
    mask_tensor (torch.Tensor): 마스크 텐서 (H, W)
    save_path (str): 저장할 이미지 경로와 파일명
    """

    # 마스크 텐서를 numpy 배열로 변환
    mask_np = mask_tensor.numpy()

    # 값의 범위를 0~255로 확장
    mask_np = (mask_np * 255).astype(np.uint8)

    # numpy 배열을 PIL 이미지로 변환
    mask_image = Image.fromarray(mask_np)

    # 이미지 저장
    mask_image.save(save_path)
    print(f"UV 마스크가 '{save_path}'에 저장되었습니다.")


