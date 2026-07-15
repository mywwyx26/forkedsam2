part 1: obtain outside lines
 - so first, consider that i don't actually know how to GET the lines in the first place
 - i could draw them and export as png? like i do with medibang???
 - however this should also not be my concern bc then there's the whole coding part
 - and there's no way it's actually hard to get automatically
 - another idea: 5 point curve so that i can match up the midpoint and stuff
 - will use napari gui the same way ezcalcium does probably (not 5 point but can draw lines)
DONE: manually drawn with napari and smoothed and saved as npy file

part 2: get inside lines, the layers and columns
 - basically just need to get one of them first, then the other can be done by perpendicular
 - i'm sure there's a code to actually get the perpendicular one at every point...
 - problem is how to get the first parallel line
 - i assume it would be like, at x percent of the outside line for both lines, this point on the inside line
   makes it y percent of the way from the first line to the second line, move along x while keeping y the same
 - also this line will come to an abrupt stop at the end lines (the perpendicular ones)
 - an issue is that sometimes the recording cuts off the ends weirdly, so it's hard to draw end lines for columns
 - this may be solved by instead making the layer lines longer, and just drawing column line in the middle
 - there will be extra bits on the ends but that should be fine, and may be hard to continue layer blindly
DONE: had claude code do this and they look pretty good
 
part 3: convert to a form sam2 can read
 - the lines can be either masks or multiple points, worth trying both
 - update: since some lines go out of frame, just do points
DONE: yes this happened, still working on negative lines
UPDATE: masks is much better than points, but negative points are helping

part 4: filter out the correct masks
 - am stumped on this one.
 - i could say get rid of the ones that are 95%+ the same?
 - but that doesn't guarantee the remaining ones are right
UNFINISHED: stuck at thresholds where same shapes don't combine, but many just disappear

bonus: better ui
 - implement cli
 - main file that runs everything
DONE: main file is good enough i don't need cli

up next: add columns
 - the edges are cut off which will make the lines harder to draw
 - they are also harder to see in general, so hard to draw midline too
 - even if i started at the midline, i would need some algorithm to extend more lines
 - once i figure out how to make the lines, running sam2 should be exactly the same