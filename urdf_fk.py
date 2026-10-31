"""Minimal URDF forward kinematics (numpy only) - used as an independent check of the MJCF."""
import numpy as np, xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

def _T(xyz, rpy):
    T = np.eye(4); T[:3,:3] = R.from_euler('xyz', rpy).as_matrix(); T[:3,3] = xyz; return T

class URDFChain:
    def __init__(self, path, tip_link, base_link='base_link'):
        root = ET.parse(path).getroot()
        joints = {}
        for j in root.findall('joint'):
            o = j.find('origin'); a = j.find('axis')
            joints[j.find('child').get('link')] = dict(
                name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
                T=_T([float(x) for x in o.get('xyz','0 0 0').split()], [float(x) for x in o.get('rpy','0 0 0').split()]),
                axis=np.array([float(x) for x in a.get('xyz').split()]) if a is not None else np.array([0,0,1.]))
        chain, link = [], tip_link
        while link != base_link:
            chain.append(joints[link]); link = joints[link]['parent']
        self.chain = chain[::-1]
        self.joint_names = [j['name'] for j in self.chain if j['type'] == 'revolute']
    def fk(self, q):
        T = np.eye(4); k = 0
        for j in self.chain:
            T = T @ j['T']
            if j['type'] == 'revolute':
                T = T @ np.block([[R.from_rotvec(j['axis']*q[k]).as_matrix(), np.zeros((3,1))],[np.zeros((1,3)),1]]); k += 1
        return T
