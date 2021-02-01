ffmpeg -r 60 -f image2 \
-i "./n9v2/frame%06d.png" \
-vcodec libx264 -crf 25 -vcodec libx264 -y -an video.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
