import bpy
from pygltflib import GLTF2
import shutil
import os

# glb 파일에 텍스처 적용 후 fbx로 변환
def apply_texture2fbx(input_glb_path, texture_path, output_fbx_path):
    # 현재 씬의 모든 객체 삭제(초기화)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # GLB 파일 불러오기
    bpy.ops.import_scene.gltf(filepath=input_glb_path)

    # 활성화된 오브젝트 가져오기 (GLB 파일에서 불러온 오브젝트)
    obj = bpy.context.selected_objects[0]

    # 재질(Material) 생성
    mat = bpy.data.materials.new(name="Material_with_Texture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    # 텍스처 노드 추가
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)

    # 텍스처 노드를 BSDF에 연결
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # 오브젝트에 재질 할당
    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # FBX로 내보내기
    bpy.ops.export_scene.fbx(filepath=output_fbx_path, path_mode='COPY', embed_textures=True)
    print(f"FBX 파일로 내보내기 완료: {output_fbx_path}")

'''
# 사용 예시
input_glb = "app/database/glb_files/1_catMango.glb"
texture_file = "app/database/output_textures/texure_12345.png"
output_fbx = "app/database/fbx_files.fbx"

apply_texture2fbx(input_glb, texture_file, output_fbx)'''

#단순 glb 파일을 fbx로 변환
def convert_glb2fbx(input_glb_path, output_fbx_path):
    # 현재 씬의 모든 객체 삭제
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # GLB 파일 불러오기
    bpy.ops.import_scene.gltf(filepath=input_glb_path)
    
    # FBX 파일로 내보내기
    bpy.ops.export_scene.fbx(filepath=output_fbx_path)
    print(f"GLB 파일이 FBX 파일로 변환되었습니다: {output_fbx_path}")

# glb에서 텍스처 추출
def extract_textures_from_glb(glb_file, output_dir):
    # GLB 파일을 불러오기
    gltf = GLTF2().load(glb_file)

    # 텍스처가 저장된 경로를 확인
    for image in gltf.images:
        uri = image.uri

        # uri가 None이 아닌 경우에만 처리
        if uri is not None:
            texture_file = os.path.join(output_dir, uri)
        
            # 텍스처 파일을 출력 디렉토리로 복사
            if os.path.exists(texture_file):
                print(f"텍스처 파일 복사: {texture_file}")
                shutil.copy(texture_file, output_dir)
            else:
                print(f"텍스처 파일을 찾을 수 없습니다: {uri}")
        else:
            print(f"이미지의 URI가 없습니다: {image}")


