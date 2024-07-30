import cv2
from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner

def main():
    video_frames=read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('model/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]



    output_video_frames=tracker.draw_annotations(video_frames,tracks,None)
    save_video(output_video_frames,'output_videos/output_video.avi')


if __name__=='__main__':
    main()







#for cropping the images 
   # for trackid,player in tracks['players'][0].items():
    #    bbox=player['bbox']
     #   frame=video_frames[0]

      #  cropped_image=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
       # cv2.imwrite(f'output_videos/cropped_image.jpg',cropped_image)
        #break    