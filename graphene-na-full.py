#!/usr/bin/python
import numpy as np
import scipy as sp

# Define constants
n_graphene = 1008
n_solute = 1   # number of solute atoms, 1 for sodium, 3 for water
n_snapshots = 20000 # number of structures in file
n_solvatom = 9 # number of atoms per solute molecule
q_solute = np.array([1.0]) # array for the solute charge(s) from OPLS, order needs to match structure
#  note!  python calls indices starting with 0, first entry is q_solvent[0]

# open file
filename = 'naetoh.gro' #change for different files
gro = open(filename,"r")

# output arrays
NaGraDist=np.zeros((n_snapshots,2)) # time trace of Na distance from graphene
ep=np.zeros((101)) # electrostatic potential array
ef=np.zeros((101)) # electrostatic field array
epns=np.zeros((101)) # electrostatic potential array - no solvent contribution
efns=np.zeros((101)) # electrostatic field array - no solvent contribution

# evaluate each snapshot
for x in xrange(n_snapshots):
  print('snapshot',x+1)
  # start by reading in coordinates and putting into numpy arrays
  timestamp = gro.readline()
  tempstring = gro.readline()
  n_atoms=int(tempstring)
  n_solvent = n_atoms - n_solute - n_graphene
  n_solv_mol = n_solvent/n_solvatom
  r_solvent = np.zeros((n_solvent,3))
  q_solvent = np.zeros((n_solvent))
  r_graphene = np.zeros((n_graphene,3))
  r_solute = np.zeros((n_solute,3))
  for y in xrange(n_solv_mol): # populate a full q_solvent
    itmp=9*y
    q_solvent[itmp]=-0.18
    q_solvent[itmp+1]=0.06
    q_solvent[itmp+2]=0.06
    q_solvent[itmp+3]=0.06
    q_solvent[itmp+4]=0.145
    q_solvent[itmp+5]=0.06
    q_solvent[itmp+6]=0.06
    q_solvent[itmp+7]=-0.683
    q_solvent[itmp+8]=0.418
  for y in xrange(n_graphene): # read in graphene
    templine = gro.readline()
#    print(templine,x,y,'graphene loop')  
    atom_res,atom,atom_ind,rx,ry,rz = templine.split()  
    r_graphene[y,:]= [float(rx),float(ry),float(rz)]
  for y in xrange(n_solute): # and the solute
    templine = gro.readline()
#    print(templine,x,y,'solute loop')   
    atom_res,atom,atom_ind,rx,ry,rz = templine.split()  
    r_solute[y,:]=[float(rx),float(ry),float(rz)]
  for y in xrange(n_solvent): # and the solvent
    templine = gro.readline()
#    print(templine,x,y,'solvent loop')    
    atom_res,atom,atom_ind,rx,ry,rz = templine.split()  
    r_solvent[y,:]=[float(rx),float(ry),float(rz)]
  templine = gro.readline()
#  print(templine,x,y,'box line')
  bx,by,bz = templine.split() # box vectors
  bx=float(bx)
  by=float(by)
  bz=float(bz)

  # Now we define the best fit plane for the graphene atoms
  m=np.ones((4,n_graphene))
  m[0,:]=r_graphene[:,0]  
  m[1,:]=r_graphene[:,1]  
  m[2,:]=r_graphene[:,2]  

#  print(m)
  u,s,v=np.linalg.svd(m,full_matrices=1,compute_uv=1)
  u[:,3]=u[:,3]/u[2,3]
  A=u[0,3]
  B=u[1,3]
  C=u[2,3]
  D=u[3,3]

#  print("normal vector, A,B,C,D, scaled by C")
#  print(u[:,3])
  # You can plot this plane in Matlab!
  # Use the following commands:
  #   [x y] = meshgrid(-5:0.5,5);
  #   z = -D - A*x - B*Y;
  #   mesh(x,y,z)
  #
  # The normal vector is conventiently scaled by the Z coefficient for this reason

  # Next - Find the point on the plane closest to the solute
  # Center so that point of the plane is at (0,0,0)
  # Rotate so that the sodium ion is at (0,0,z)
  # Equidistant rings on the plane surface then become trivial

  # Find the closest point in the plane
  dist = A*r_solute[0,0]+B*r_solute[0,1]+C*r_solute[0,2]+D
  scalar = np.sqrt(A*A+B*B+C*C)
  dist = dist/scalar
  A1=dist*A/scalar
  B1=dist*B/scalar
  C1=dist*C/scalar
  X1=r_solute[0,0]-A1
  Y1=r_solute[0,1]-B1
  Z1=r_solute[0,2]-C1
#  print('should be zero',A*X1+B*Y1+C*Z1+D)
  # Center the solute and solvent
  r_solute[0,0]=r_solute[0,0]-X1
  r_solute[0,1]=r_solute[0,1]-Y1
  r_solute[0,2]=r_solute[0,2]-Z1
  r_solvent[:,0]=r_solvent[:,0]-X1
  r_solvent[:,1]=r_solvent[:,1]-Y1
  r_solvent[:,2]=r_solvent[:,2]-Z1
  # Find the rotation matrix
  scalar=r_solute[0,0]**2+r_solute[0,1]**2+r_solute[0,2]**2
  scalar=np.sqrt(scalar)
  X2=r_solute[0,0]/scalar
  Y2=r_solute[0,1]/scalar
  Z2=r_solute[0,2]/scalar
#  print('should be 1',np.sqrt(X2**2+Y2**2+Z2**2))
  mm=[X2,Y2,Z2]
  nn=[0,0,1]
  vv=np.cross(mm,nn)
  ss=np.linalg.norm(vv)
  uu=np.dot(mm,nn)
  vx=[[0,-vv[2],vv[1]],[vv[2],0,-vv[0]],[-vv[1],vv[0],0]]
  vx2=np.dot(vx,vx)
  ii=np.identity(3)
  rot=ii+vx+(1-uu)/(ss**2)*vx2
  # Rotate the solute and solvent coordinates
  w=[r_solute[0,0],r_solute[0,1],r_solute[0,2]]
  ww=np.dot(rot,w)
  r_solute[0,0]=ww[0]
  r_solute[0,1]=ww[1]
  r_solute[0,2]=ww[2]
  outline=x,r_solute
  NaGraDist[x,0]=x
  NaGraDist[x,1]=r_solute[0,2]
  for y in xrange(n_solvent): # and the solvent
    w=[r_solvent[y,0],r_solvent[y,1],r_solvent[y,2]]
    ww=np.dot(rot,w)
    r_solvent[y,0]=ww[0]
    r_solvent[y,1]=ww[1]
    r_solvent[y,2]=ww[2]


  # start recording q/r (potential) and q/r**2 (field) values
  # sample in polar coordinates, 10 degree increments and 0.01 nm radial values
  ep=np.zeros((101))
  ef=np.zeros((101))
  for y in xrange(101):
    rad=y/100.0
    for z in xrange(12):
      theta=2.0*np.pi*z/12.0
      xtmp=rad*np.cos(theta) # sampling in polar space, but using cartesian for calcs
      ytmp=rad*np.sin(theta)
      rtmp=np.linalg.norm([r_solute[0,0]-xtmp,r_solute[0,1]-ytmp,r_solute[0,2]])
      ep[y]=ep[y]+q_solute[0]/rtmp # sodium contribution
      ef[y]=ef[y]+q_solute[0]/(rtmp**2) # sodium contribution
      epns[y]=epns[y]+q_solute[0]/rtmp # sodium contribution - no solv
      efns[y]=efns[y]+q_solute[0]/(rtmp**2) # sodium contribution - no solv
      for xx in xrange(n_solvent):
        rtmp=np.linalg.norm([r_solvent[xx,0]-xtmp,r_solvent[xx,1]-ytmp,r_solvent[xx,2]])
        ep[y]=ep[y]+q_solvent[xx]/rtmp # solvent contribution
        ef[y]=ef[y]+q_solvent[xx]/(rtmp**2) # solvent contribution

print('writing output')          
ep=ep/(12*n_snapshots) # average
ef=ef/(12*n_snapshots) # average 
epns=epns/(36*n_snapshots) # average
efns=efns/(36*n_snapshots) # average 

np.savetxt('NaGraDist.dat',NaGraDist)
np.savetxt('ElecPot.dat',ep)
np.savetxt('ElecField.dat',ef)
np.savetxt('ElecPot-NoSolv.dat',epns)
np.savetxt('ElecField-NoSolv.dat',efns)





