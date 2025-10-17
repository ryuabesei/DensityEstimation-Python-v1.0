from sgp4.api import Satrec, jday

#ISSのTLE
s = '1 25544U 98067A   21226.49389238  .00001429  00000-0  34174-4 0  9998'
t = '2 25544  51.6437  54.3833 0001250 307.1355 142.9078 15.48901431297630'

#SGP4による位置・速度の計算
satellite = Satrec.twoline2rv(s, t)
jd, fr = jday(2021, 8, 15, 0, 0, 0)
e, r, v = satellite.sgp4(jd,fr)

print(r)
print(v)

