import numpy as np

def computeNormalVectorTriangulation(XYZ, TRI, strPosition):

    """
     This function takes as input a 2D unrestricted triangulation and computes
     the normal vector of the surface.
    
     Input :
               "XYZ" is the coordinate of the vertex of the triangulation (nx3 matrix).
               "TRI" is the list of triangles which contain indexes of XYZ (mx3 matrix).
               "strPosition" is the position where the normal is computed. It
               could be 'center-cells' for a computation on the center of each
               triangle or could be 'vertices' and the vectors are computed at
               vertices with respect to the neighbour cells (string).
    
     Output :
               "NormalVx", "NormalVy" and "NormalVz" are the component of
               normal vectors (normalized to 1).
               "PosVx", "PosVy" and "PosVz" is the positions of each vector.
     
     Note : 
               if strPosition == 'center-cells', then the dimension of each
               output are mx1.
               if strPosition == 'vertices', then the dimension of each
               output are nx1. 
    
               All cells have to be enumerated clockwise or counter-clock.
    
     Example :
    
     [X,Y,Z]=peaks(25)
     X=reshape(X,[],1)
     Y=reshape(Y,[],1)
     Z=0.4*reshape(Z,[],1)
     TRI = delaunay(X,Y)
     [NormalVx NormalVy NormalVz PosVx PosVy PosVz]=computeNormalVectorTriangulation([X Y Z],TRI,'vertices')
    
     quiver3(PosVx,PosVy, PosVz, NormalVx, NormalVy, NormalVz), axis equal
     hold on
     trimesh(TRI,X,Y,Z)
    
     David Gingras, February 2009 
    """

    # convert index in 0-based for NumPy    
    TRI0 = TRI - 1
    v1 = XYZ[TRI0[:, 2], :] - XYZ[TRI0[:, 1], :]
    v2 = XYZ[TRI0[:, 1], :] - XYZ[TRI0[:, 0], :]
    NormalTri = np.cross(v1, v2)

    norms = np.linalg.norm(NormalTri, axis=1, keepdims=True)  # (T,1)
    if len(norms[norms == 0]):
        print('zero norm triangle')
    norms[norms == 0] = 1.0
    NormalTri = - NormalTri / norms

    if strPosition.lower() == 'center-cells':
        NormalVx, NormalVy, NormalVz = NormalTri[:, 0], NormalTri[:, 1], NormalTri[:, 2]
        PosVx, PosVy, PosVz = centerTri3D(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], TRI)        
        
    elif strPosition.lower() == 'vertices':
        invTRI = buildInverseTriangulation(TRI)
        N = XYZ.shape[0]       
        NormalVx = np.zeros(N)        
        NormalVy = np.zeros(N)        
        NormalVz = np.zeros(N)
        for j in range(N):
            NormalVx[j] = NormalTri[removeD0(invTRI[j, :])-1, 0].mean()                
            NormalVy[j] = NormalTri[removeD0(invTRI[j, :])-1, 1].mean()                
            NormalVz[j] = NormalTri[removeD0(invTRI[j, :])-1, 2].mean()

        PosVx, PosVy, PosVz = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]     
    else:
        raise ValueError("The third argument input is not correct. Use 'center-cells' or 'vertices'.")
    
    return NormalVx, NormalVy, NormalVz, PosVx, PosVy, PosVz
    

def buildInverseTriangulation(TRI):
    # Building the inverse triangulation, i.e. a link from node indexes to
    # triangle indexes.
    nbTri = len(TRI)
    nbNode = np.max(TRI.reshape(-1, 1))
    comp = np.zeros(nbNode, 1)
    invTRI = np.zeros(nbNode, 8)

    for i in range(nbTri):               
        for j in range(3):                      
            index = TRI[i, j] - 1           
            comp[index] += 1                   
            invTRI[index, comp[index]-1] = i + 1
        return invTRI

def centerTri3D(X, Y, Z, tri1):
    # This function return the position of the center of a cells
    tri0 = tri1 - 1
    Xc = X[tri0].mean(axis=1)
    Yc = Y[tri0].mean(axis=1)
    Zc = Z[tri0].mean(axis=1)
    return Xc, Yc, Zc

def removeD0(x):
    # Removing duplicate and null values
    s = np.sort(x)
    s1 = np.concatenate(([0], s))              
    s2 = np.concatenate((s, [s[len(s)-1]])) 
    ds = s1 - s2
    out = s[(ds != 0)[1:]]     
    return out

