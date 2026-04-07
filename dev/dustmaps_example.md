### Example from the dustmaps docs
For our first example, let’s load the Schlegel, Finkbeiner & Davis (1998) – or “SFD” – dust reddening map, and then query the reddening at one location on the sky:

```
from __future__ import print_function
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

coords = SkyCoord('12h30m25.3s', '15d15m58.1s', frame='icrs')
sfd = SFDQuery()
ebv = sfd(coords)

coords = SkyCoord('12h30m25.3s', '15d15m58.1s', frame='icrs')
print('E(B-V) = {:.3f} mag'.format(ebv))

>>> E(B-V) = 0.030 mag
```
A couple of things to note here:

In this example, we used from __future__ import print_function in order to ensure compatibility with both Python 2 and 3.
Above, we used the ICRS coordinate system, by specifying frame=’icrs’.
SFDQuery returns reddening in a unit that is similar to magnitudes of E(B-V). However, care should be taken: a unit of SFD reddening is not quite equivalent to a magnitude of E(B-V). The way to correctly convert SFD units to extinction in various broadband filters is to use the conversions in Table 6 of Schlafly & Finkbeiner (2011).