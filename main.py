fname = "name.wav"
s = f"sudo arecord -D hw:1,0 -d 3 -r 44100 -f S16_LE {fname}"
