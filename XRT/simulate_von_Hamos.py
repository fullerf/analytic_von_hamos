# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.oes as roe

showIn3D = False
nrays = 5e5
radius = 500
p = 0-radius
r = 500
c = -500
a = 0-radius
length = -50, 50
central_E = (9130,9140,)
if len(sys.argv)>1:
    p = -(float(sys.argv[1]) - radius)
if len(sys.argv)>2:
    r = float(sys.argv[2])
if len(sys.argv)>3:
    a = -(float(sys.argv[3]) - radius)
if len(sys.argv)>4:
    c = float(sys.argv[4])
if len(sys.argv)>5:
    central_E = (float(sys.argv[5]),)
if len(sys.argv)>6:
    nrays = sys.argv[6]

fout = f"output_at_p={sys.argv[1]}_r={sys.argv[2]}_a={sys.argv[3]}_c={sys.argv[4]}_e={sys.argv[5]}.pickle"
if os.path.isfile(f"./{fout}"):
    os.remove(f"./{fout}")

print('p in xrt is ',p)
print('r in xrt is ',r)
print('a in xrt is ',a)
print('c in xrt is ',c)

crystalMaterial = 'Si'
if crystalMaterial == 'Si':
    d111 = 3.1354161
elif crystalMaterial == 'Ge':
    d111 = 3.2662725
else:
    raise
    
class CrystalSi(rm.CrystalDiamond):
    def __init__(self, hkl):
        self.a = 5.4307717932001225
        self.d = np.sqrt(self.a**2/(hkl[0]**2 + hkl[1]**2 + hkl[2]**2))
        super().__init__(hkl, self.d, elements='Si')

crystal = CrystalSi((4,4,0))


class Cylinder(roe.SurfaceOfRevolution):
    def __init__(self, *args, **kwargs):
        self.r = kwargs.pop('r')
        super(Cylinder, self).__init__(*args, **kwargs)

    def local_r(self, s, phi):
        return self.r * np.ones_like(s)

    def local_n(self, s, phi):
        a = -np.sin(phi)
        b = np.zeros_like(s)
        c = -np.cos(phi)
        return a, b, c


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.source = rs.GeometricSource(
        beamLine, 'GeometricSource', (a, c, 0), nrays=nrays,
        distx=None, distz=None,
        distxprime='annulus', dxprime=(0.85, 1.15),
        energies=central_E,
        polarization=None)

    beamLine.cylinder = Cylinder(
        beamLine, center=[0, 0, 0], r=radius,
        limPhysX=[0, radius*2],  # try with and without
        limPhysY=[-50, 50],
        material=crystal)
        

    beamLine.screen = rsc.Screen(beamLine, center=[p, r, 0],
                                 x=(0,1,0), z=(0,0,1))
    return beamLine


def run_process(bl):
    beamSource = bl.source.shine()
    beamCylinderGlobal, beamCylinderLocal = bl.cylinder.reflect(beamSource)
    beamScreen = bl.screen.expose(beamCylinderGlobal)
    outDict = {'beamSource': beamSource,
               'beamCylinderGlobal': beamCylinderGlobal,
               'beamCylinderLocal': beamCylinderLocal,
               'beamScreen': beamScreen}
    if showIn3D:
        bl.prepare_flow()
    return outDict
raycing.run.run_process = run_process


def define_plots(bl):
    plots = []

    plot = xrtp.XYCPlot(
        beam='beamCylinderLocal', aspect='auto',
        xaxis=xrtp.XYCAxis(label='phi', unit='rad', fwhmFormatStr='%.3f'),
        yaxis=xrtp.XYCAxis(label='s', fwhmFormatStr='%.3f'))
    plot.ax2dHist.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2, np.pi))
    plot.ax2dHist.set_xticklabels(
        (r'-$\pi$', r'-$\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'))
    plot.yaxis.limits = [-20, 20]
    plot.xaxis.limits = [-20, 20]
    plots.append(plot)

    plot = xrtp.XYCPlot(
        beam='beamScreen', aspect='auto',
        xaxis=xrtp.XYCAxis(label='x', fwhmFormatStr='%.3f'),
        yaxis=xrtp.XYCAxis(label='z', fwhmFormatStr='%.3f'),
        persistentName=fout)
    plots.append(plot)
    plot.xaxis.limits = [-20,20]
    plot.yaxis.limits = [-20,20]

    return plots

def main():
    bl = build_beamline()
    if showIn3D:
        bl.glow(scale=[6., 6., 6.])
        return
    plots = define_plots(bl)
    xrtr.run_ray_tracing(beamLine=bl, repeats=1, plots=plots)


if __name__ == '__main__':
    main()
