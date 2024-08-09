from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .forms import BasketballGameForm
from .models import BasketballGame
import cv2
import os
from . import utils
import json
import numpy as np
import random

def home_page(request):
    return render(request, 'court_detection/home.html')


def upload_video(request):
    if request.method == 'POST':
        form = BasketballGameForm(request.POST, request.FILES)
        if form.is_valid():
            game = form.save()
            fx, fy, cx, cy, k1, k2, k3, k4 = request.POST.get('fx'), request.POST.get('fy'), request.POST.get('cx'), request.POST.get('cy'), request.POST.get('k1'), request.POST.get('k2'), request.POST.get('k3'), request.POST.get('k4')
            game.distortion_parameters = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'k4': k4
            }
            game.save()
            return redirect('select_corners', game_id=game.id)
        else:
            print(form.errors)
    else:
        form = BasketballGameForm()
    return render(request, 'court_detection/upload_video.html', {'form': form})

def select_corners(request, game_id):
    game = get_object_or_404(BasketballGame, id=game_id)
    video_path = os.path.join(settings.MEDIA_ROOT, game.video.name)

    # Estrai un frame in posizione random
    cap = cv2.VideoCapture(video_path)
    n = random.randint(0, cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    cap.release()

    #if not ret:
    #    return render(request, 'yourapp/error.html', {'message': 'Could not read video frame.'})

    # Salva il frame come immagine
    if game.distortion_parameters['fx']:
        frame = utils.undistort_frame(frame, utils.camera_matrix(game.distortion_parameters), utils.dist_coeffs(game.distortion_parameters))
    
    frame_path = os.path.join(settings.MEDIA_ROOT, 'frames', f'frame_{game_id}.jpg')
    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    cv2.imwrite(frame_path, frame)

    frame_url = os.path.join(settings.MEDIA_URL, 'frames', f'frame_{game_id}.jpg')

    return render(request, 'court_detection/select_corners.html', {'game': game, 'frame_url': frame_url})

@csrf_exempt
def save_corners(request, game_id):
    if request.method == 'POST':
        game = get_object_or_404(BasketballGame, id=game_id)
        try:
            data = json.loads(request.body)
            corners = data.get('corners')
        
            if not corners or len(corners) != 6:
                return JsonResponse({'status': 'error', 'message': f'Invalid number of corners. Expected 6, got {len(corners)}'})
            
            cap = cv2.VideoCapture(game.video.path)
            ret, frame = cap.read()
            cap.release()
            h, w = frame.shape[:2]
            #Multiply each x corner by the width of the frame and each y corner by the height of the frame
            corners = [{'x': int(corner['x'] * w), 'y': int(corner['y'] * h)} for corner in corners]
            game.corners = utils.get_corners(corners)
            
            game.save()
            
            return JsonResponse({'status': 'success'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'})
        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def top_view(request, game_id):
    game = get_object_or_404(BasketballGame, id=game_id)
    video_path = game.video.path
    output_path = os.path.join(settings.MEDIA_ROOT, 'top_views', f'top_view_{game_id}.jpg')
    output_path_undistort_frame = os.path.join(settings.MEDIA_ROOT, 'top_views', f'undistorted_{game_id}.jpg')
    output_path_distorted_frame = os.path.join(settings.MEDIA_ROOT, 'top_views', f'distorted_{game_id}.jpg')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_path = os.path.join(settings.MEDIA_ROOT, 'frames', f'frame_{game_id}.jpg')
    frame = cv2.imread(frame_path)

    top_left = game.corners['P0']
    bottom_right = game.corners['P11']
    bottom_left = game.corners['P3']
    top_right = game.corners['P8']
    
    middle_top = game.corners['P6']
    middle_bottom = game.corners['P7']
    #print("middle_top", middle_top)
    #print("middle_bottom", middle_bottom)
#
    #print("top_left", top_left)
    #print("bottom_right", bottom_right)
    #print("bottom_left", bottom_left)
    #print("top_right", top_right)

    undistort_frame = frame
    
    #utils.draw_points(undistort_frame, top_left, bottom_right, bottom_left, top_right, middle_top, middle_bottom)

    points = np.array([
        [top_left["x"], top_left["y"]],
        [top_right["x"], top_right["y"]],
        [bottom_left["x"], bottom_left["y"]],
        [bottom_right["x"], bottom_right["y"]],
        [middle_top["x"], middle_top["y"]],
        [middle_bottom["x"], middle_bottom["y"]],
    ], dtype=np.float32)
    points_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle_top', 'middle_bottom']

    points = points.reshape(-1, 1, 2)

    #print("Salvo undistorted frame in: ", output_path)
    #cv2.imwrite(output_path_undistort_frame, undistort_frame)

    h, w = undistort_frame.shape[:2]
    h, w = h // 2, w // 2
    dst = np.array([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h],
        [w // 2, 0],
        [w // 2, h],
    ], dtype=np.float32)
    M, status = cv2.findHomography(points, dst)
    print("Points used for the homography: ")
    for name, used in zip(points_names, status):
        if used:
            print(name)
    frame_top_view = cv2.warpPerspective(undistort_frame, M, (w, h))

    utils.draw_lines(frame_top_view)

    #print("Salvo top view in: ", output_path)
    cv2.imwrite(output_path, frame_top_view)
    output_url = os.path.join(settings.MEDIA_URL, 'top_views', f'top_view_{game_id}.jpg')
    return render(request, 'court_detection/top_view.html', {'game': game, 'top_view_url': output_url})