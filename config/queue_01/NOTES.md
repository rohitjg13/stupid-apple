# config/queue_01 — the screen-recorded queue clip

`footage/queue_01.mp4`, 1312x740, 30 fps, 45 s: an overhead view of a
single-file queue at a hot-food counter. The service point is on the right,
so **cell 0 is the rightmost cell** and cell 7 is the back of the queue.

Lane cells are authored at PL scale (320x240), as the register map requires.
The letterbox from source to PL is:

    x_pl = x_src * 0.2439
    y_pl = y_src * 0.2439 + 29.5      # 1312x740 fits on width; 59 px of bar
                                      # top and bottom at 640x480, so 29.5 at PL

The band is y = 80..150. It stops short of y = 65 on purpose: the seating area
across the top of the frame has people walking through it, and a taller cell
would count them as queuers. It does still catch the legs of anyone who walks
right up to the rail, which is the main source of false positives here.

One counter (`checkout.counters: 1`) — there is a single server.

The `shelf` stream is bound to the same file so the clipset validates; only
`--streams overhead` is meant to be run against it. The single ROI is the hot
food display and exists so planogram.json has a facing to reference.
