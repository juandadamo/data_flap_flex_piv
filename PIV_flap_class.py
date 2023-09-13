#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import scipy.io as scio
from scipy import interpolate
import pylab, socket
import numpy as np
import matplotlib.pyplot as plt
import glob,os
import sys
sys.path.insert(0, '../sistdinamicos')
#import function_clustering_m as fc
from scipy.signal import correlate2d as corr2
from scipy.signal import medfilt,medfilt2d
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from scipy.signal import medfilt,medfilt2d, correlate2d
from skimage.morphology import closing, square, disk
from skimage.measure import label
#from matplotlib2tikz import save as tikz_save
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 16;
mpl.rcParams['xtick.labelsize'] = 16;
mpl.rcParams['ytick.labelsize'] = 16;
mpl.rcParams['font.family'] = 'serif';
mpl.rcParams['axes.labelsize'] = 16
#mpl.rcParams['mathtext.cal']=
import ReadIM
import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class PIV_ala:
    def __init__(self, name_file):
      self.name_file = name_file
      self.name = name_file.split('/')[-1] 
      self.dirname = '/'.join(name_file.split('/')[:-1]) +'/'
      self.dir_cluster = self.dirname.replace('VEC_Python2','salida4')
      self.dir_out = self.dirname.replace('VEC_Python2','salida_f')
      self.dir_obj = self.dirname.replace('VEC_Python2','salida_obj')
      self.dir_images = self.dirname.replace('data_juan/2017/flap_PIV/VEC_Python2/','data_manu/Manu/')
      if os.path.isdir(self.dir_out) == False:
            os.mkdir(self.dir_out)
            os.mkdir(self.dir_out+'/cluster')
      self.freq = np.float(self.name.split('f')[-1])
      self.listing_images()
      self.x_stabhidro = 0
      self.xi_stabidro = 0
      self.omegas = 0

    def listing_images(self):
      #dir_images = '/media/juan/data_manu/Manu/'
      dir_images = self.dir_images
      lista_files1 = os.listdir(dir_images)
      lista_files1 = glob.glob(dir_images+'*')
      lista_files2 = [os.path.isdir(listafilesi) for listafilesi in lista_files1]
      lista_images_dirs1 = np.asarray(lista_files1)[lista_files2][:-2]
      lista_images = []
      for listai in lista_images_dirs1:
          lista_files1 = glob.glob(listai+'/*')
          index_lista_files1 = [os.path.isdir(listafilesi) for listafilesi in lista_files1]
          lista_images_dirs1 = np.asarray(lista_files1)[index_lista_files1]
          lista_images.append(lista_images_dirs1)
      lista_images = [item for sublist in lista_images for item in sublist]
      
      matching1 = [s for s in lista_images if self.name[2:6] in s]
      matching2 = [s for s in matching1 if self.name[-3:] in s]
      
      self.lista_images = matching2
      casoi = self.lista_images[0][:-2]
      self.caso_imagen = casoi
      casos = np.sort(glob.glob(casoi+'*'))  
      casos2 = [os.path.isdir(casoi) for casoi in casos]
      self.caso_imagen_s = casos[casos2]
      archivos_i = []
      for i,dir_i in enumerate(self.caso_imagen_s):
        archivos_i.append(np.sort(glob.glob(dir_i+'/*.im7')))
      archivos_i = np.asarray([item for sublist in archivos_i for item in sublist])
      self.archivos_imagen = archivos_i
    def carga_imagen(self,i):
      filei = self.archivos_imagen[i]
      image_array = np.zeros((400,350))
      xc,yc = [307,619]
      xL,yL = [590,619]
      Lpixel = xL - xc
      nx1, ny2 = [300,yc+200]
      m,n = image_array.shape
      nx2,ny1 = (nx1+n,ny2-m)
      Lpixel = xL - xc

      x,y = [self.x,self.y]
      Lcuerda = 35.
      escala_mm = Lcuerda/Lpixel  #mm/px
      escala = 1./Lpixel

      xc1 = (xc - 300)/(1.*Lpixel)
      yc1 = (yc - 418)/(1.*Lpixel)
      self.ximages=np.arange(n)*escala-xc1
      self.yimages=(np.arange(m)*escala-yc1)
      
      
      vbuff, vatts = ReadIM.extra.get_Buffer_andAttributeList(filei)
      v_array, vbuff = ReadIM.extra.buffer_as_array(vbuff)
      im1 = v_array[0][ny1:ny2][:,nx1:nx2]
      im2 = v_array[1][ny1:ny2][:,nx1:nx2]
      self.image_i,self.image_i2 = [im1[::-1],im2[::-1]]
      #self.image_i,self.image_i2 = [im1,im2]
      self.images_xc = xc
      self.images_yc = yc
      self.images_yL = yL
      self.images_xL = xL
      return
      
      
    def carga_imagenes(self):
      casoi = self.lista_images[0][:-2]
      self.caso_imagen = casoi
      casos = np.sort(glob.glob(casoi+'*'))  
      casos2 = [os.path.isdir(casoi) for casoi in casos]
      self.caso_imagen_s = casos[casos2]
      archivos_i = []
      for i,dir_i in enumerate(self.caso_imagen_s):
        archivos_i.append(np.sort(glob.glob(dir_i+'/*.im7')))
      archivos_i = np.asarray([item for sublist in archivos_i for item in sublist])
      self.archivos_imagen = archivos_i
      set_images = np.zeros((680,400,350))
      xc,yc = [307,619]
      xL,yL = [590,619]
      Lpixel = xL - xc
      nx1, ny2 = [300,yc+200]
      M,m,n = set_images.shape
      nx2,ny1 = (nx1+n,ny2-m)
      
      im_secuencia = []
      for i,filei in enumerate(archivos_i):
        vbuff, vatts = ReadIM.extra.get_Buffer_andAttributeList(filei)
        v_array, vbuff = ReadIM.extra.buffer_as_array(vbuff)
        set_images[i]=v_array[0][ny1:ny2][:,nx1:nx2]
      self.images = set_images
      self.images_xc = xc
      self.images_yc = yc
      self.images_yL = yL
      self.images_xL = xL
      return 
    def binarize_im(self):
        images_t = np.copy(self.images)
        image_mean = np.copy(images_t[0])
        image_std = np.copy(images_t[0])
        N,m,n = images_t.shape
        for imagei in images_t:
            imagei[:200,:20]=0
            imagei[:,:90][imagei[:,:90]<10000]=0
            imagei[:,:90][imagei[:,:90]<15000]=0
            imagei[:,:80][imagei[:,:80]>15000]=1000
            imagei[:,90:][imagei[:,90:]>15000]=1000
            imagei[imagei>700]=1000
            imagei[imagei<1000]=0
            image_mean += imagei
            image_std += imagei**2
        self.images_bin = images_t       
    def carga_caso(self):
        L=35
        casoi = self.name_file
        casos = np.sort(glob.glob(casoi+'*'))    
        for i,casoi in enumerate(casos):
          A = np.load(casoi, encoding = 'latin1')
          x = A['x']
          y = A['y']
          v = A['v']
          if i>0:
            vy = np.vstack((vy,v['vy']))
            vx = np.vstack((vx,v['vx']))
          else:
            vx = v['vx']
            vy = v['vy']
        self.x,self.y = [x/L,y/L+.1]
        self.vx,self.vy = [vx,vy]
        self.L = L
        return x,y,vx,vy
    def stats(self):
        vx_m = self.vx.mean(0)
        vy_m = self.vy.mean(0)
        vx_std = self.vx.std(0)
        vy_std = self.vy.std(0)
        vxvy_std = ((self.vx-vx_m)*(self.vy-vy_m)).mean(0)
        Uinf = np.abs(self.vx.mean(0)[50:65,50:70].mean())
        Uinf = 1
        self.vxstd = vx_std / Uinf
        self.vystd = vy_std / Uinf
        self.vxvystd = vxvy_std / Uinf
        self.vxm = vx_m / Uinf
        self.vym = vy_m / Uinf
        self.Uinf = Uinf
        
        #self.Uinf = Uinf
        #self.vx = self.vx  / Uinf
        #self.vy = self.vy  / Uinf
        return vx_m,vy_m,vx_std,vy_std,vxvy_std,Uinf
    
    def carga_cluster2(self,graba=0):
        numeros = np.arange(len(self.archivos_imagen))
        self.carga_imagen(0)
        nameout = self.dir_out+'/cluster/'+self.name+'_cluster.npz' 
        if os.path.isfile(nameout) == True:
            B = np.load(nameout,allow_pickle=True, encoding = 'latin1')
            self.salida_vortex,self.n_orden = [B['salida_vortex'],B['n_orden']]
            filecluster = self.name_file.replace('/VEC_Python2/','/salida4/')+'_cluster.npz'
            C = np.load(filecluster,allow_pickle=True, encoding = 'latin1')
            [n_orden,indices_in,kmeans_uy_labels,salida_vortex] = [C['n_orden'],
                            C['indices_in'],C['kmeans_uy_labels'],C['salida_vortex']]
            self.indices_in = indices_in
            self.kmeans_uy_labels = kmeans_uy_labels
            self.n_orden = n_orden
            self.angle_phase = C['angle_phase']
        vx_cluster = np.tile(np.zeros_like(self.vx[0]),[12,1,1])
        vy_cluster = np.tile(np.zeros_like(self.vx[0]),[12,1,1])

        salida_xy = []
        #raise ValueError()
        for i,ni in enumerate(self.n_orden[:]):
            vx_cluster[i] = self.vx[self.indices_in[0]][self.kmeans_uy_labels == ni].mean(0)
            vy_cluster[i] = self.vy[self.indices_in[0]][self.kmeans_uy_labels == ni].mean(0)
            indice_cluster_i = np.nonzero(self.kmeans_uy_labels == ni)[0]
            numeros_cluster_i = numeros[self.indices_in[0][indice_cluster_i]]
            imagenes = np.tile(np.zeros_like(self.image_i),[len(numeros_cluster_i),1,1])
            for j,num_i in enumerate(numeros_cluster_i):
                self.carga_imagen(num_i)
                imagenes[j] = self.image_i
            imagen_f = imagenes[:1].mean(0)
            imagen2 = np.copy(imagen_f)
            imagen3 = np.copy(imagen_f)

            imagen2[:,self.ximages<0.27] = 0
            imagen3[:,self.ximages>0.27] = 0
            imagen3[:,self.ximages<0.05] = 0
            umbral1 = 1000

            imagen2[imagen2<umbral1] = 0
            imagen2[imagen2>umbral1] = 1
            imagen3[imagen3>10800] = 0
            imagen3[imagen3<9000] = 0
            imagen3[imagen3>0] = 1
            imagen3[np.abs(self.yimages).argmin(),np.abs(self.ximages).argmin()] = 1
            imagen4 = imagen2 + imagen3
            Y1 = np.nonzero([imagen4 ==1])[1]
            X1 = np.nonzero([imagen4 ==1])[2]
            del(imagenes)
            X1 = self.ximages[X1]
            Y1 = self.yimages[Y1]
            z = np.polyfit(X1,Y1,3)
            fp = np.poly1d(z)
            xs = np.linspace(X1[:].min(),X1.max(),50)
            salida_xy.append((X1,Y1,xs,fp))
        self.salida_xy = np.asarray(salida_xy)
        save_object(self,self.dir_obj+self.name+'.pkl')
        print('guardando '+self.dir_obj+self.name+'.pkl')
        return    
            
    def carga_cluster(self,graba=0):
        nameout = self.dir_out+'/cluster/'+self.name+'_cluster.npz' 
        if os.path.isfile(nameout) == True:
            B = np.load(nameout,allow_pickle=True, encoding = 'latin1')
            #self.x,self.y,self.Uinf = [B['x'],B['y'],B['Uinf']]
            #self.vx_cluster,self.vy_cluster = [B['vx_cluster'],B['vy_cluster']]
            #self.vort,self.q = [B['vort'],B['q']]
            self.salida_vortex,self.n_orden = [B['salida_vortex'],B['n_orden']]
            filecluster = self.name_file.replace('/VEC_Python2/','/salida4/')+'_cluster.npz'
            C = np.load(filecluster,allow_pickle=True, encoding = 'latin1')
            [n_orden,indices_in,kmeans_uy_labels,salida_vortex] = [C['n_orden'],
                            C['indices_in'],C['kmeans_uy_labels'],C['salida_vortex']]
            self.indices_in = indices_in
            self.kmean_labels = kmeans_uy_labels
            
        else:
            L = 35
            ang_rig = 0.18
            L_frontal = np.sin(ang_rig)*L
            x,y,vx,vy  = self.carga_caso()
            #vx = -vx
            #x,y = [x/L,y/L+.1]
            #x = x
            dx = np.abs(x[1]-x[0])
            nclust = 12
            vxm,vym,vxstd,vystd,vxvystd,Uinf2 = self.stats()
            Uinf = np.float(self.name.split('_')[0].split('=')[1])
            Uinf = vxm[60:-10,10].mean()
            vxm,vym = (vxm/Uinf,vym/Uinf)
            filecluster = self.dir_cluster+self.name+'_cluster.npz'
            B = np.load(filecluster,allow_pickle=True, encoding = 'latin1')
            [n_orden,indices_in,kmeans_uy_labels,salida_vortex] = [B['n_orden'],
                              B['indices_in'],B['kmeans_uy_labels'],B['salida_vortex']]
            #raise ValueError()
            vy_cluster,vx_cluster=([],[])
            vx,vy = (vx/Uinf,vy/Uinf)  
            vxm,vym,vxstd,vystd,vxvystd,Uinf2 = self.stats()
            for k_cluster in range(nclust):
              vx_cluster.append( vx[indices_in][0][kmeans_uy_labels==k_cluster].mean(0))
              vy_cluster.append( vy[indices_in][0][kmeans_uy_labels==k_cluster].mean(0))
            vx_cluster = np.asarray(vx_cluster)
            vy_cluster = np.asarray(vy_cluster)
            vort,s,q,dvxdx_c,dvxdy_c,dvydx_c,dvydy_c = np.tile(np.zeros_like(vx_cluster),[7,1,1,1])
            for j,vy_cluster_j in enumerate(vy_cluster):
                dvxdx,dvxdy = np.gradient(vx_cluster[j],dx)
                dvydx,dvydy = np.gradient(vy_cluster_j,dx)
                dvxdx_c[j],dvxdy_c[j],dvydx_c[j],dvydy_c[j] = dvxdx,dvxdy,dvydx,dvydy
                vort[j]= (dvydx-dvxdy)
                s[j]  = dvxdy + dvydx
                q[j] =  (dvydx-dvxdy)**2 - (dvxdy + dvydx)**2
            vxp,vyp = [vx_cluster-vx_cluster.mean(0),vy_cluster-vy_cluster.mean(0)]
            dvxvypdx,dvxvypdy = np.gradient(vxvystd**2,dx)
            dvyvypdx,dvyvypdy = np.gradient(vystd**2,dx)            
            self.x,self.y,self.Uinf,self.vx_cluster,self.vy_cluster,self.vort,self.q = [x,y,Uinf,vx_cluster,vy_cluster,vort,q]
            self.salida_vortex,self.n_orden = [salida_vortex,n_orden]
            self.indices_in = indices_in
            self.kmean_labels = kmeans_uy_labels
        if graba ==1:
           nameout = self.dir_out+'/cluster/'+self.name+'_cluster.npz' 
           dictsal = {'x':self.x,'y':self.y,'Uinf':self.Uinf,'vx_cluster':self.vx_cluster,
                      'vy_cluster':self.vy_cluster,'vort':self.vort,'q':self.q,
                     'salida_vortex':self.salida_vortex,'n_orden':self.n_orden}
           np.savez(nameout,**dictsal)
        
        return 
    def vortex_portrait(self,ver=0):
        x,y,vort,n_orden,salida_vortex = [self.x,self.y,self.vort,self.n_orden,self.salida_vortex]
        fig0,ax0 = plt.subplots(4,3,sharex=True,sharey=True,figsize= (10,7));
        ax0 = ax0.ravel()
        niveles = np.linspace(-15,15,21)
        for i,ax0i in enumerate(ax0):
            im = ax0i.contourf(x,y,vort[n_orden[i]].T,levels=niveles,extend='both')
            ax0i.grid()
            for salidai in salida_vortex[n_orden[i]]:
                if x[salidai[0]]>1.25:
                    if np.abs(salidai[2])>.1:
                        ax0i.plot(x[salidai[0]],y[salidai[1]],'w+',markersize=7)
                        if salidai[2]>0:
                            ax0i.text(x[salidai[0]]-.3,y[salidai[1]]-.2,'$\Gamma=%.2g$'%np.abs(salidai[2]))
                        else:
                            ax0i.text(x[salidai[0]]-.3,y[salidai[1]]+.2,'$\Gamma=%.2g$'%np.abs(salidai[2]))
            ax0i.set_ylim([-.6,.6])
            ax0i.set_xlim([0,x.max()])
            ax0[4].set_xlabel('$x/L$')
            ax0[4].set_ylabel('$y/L$')
        fig0.tight_layout()
        #
        cbar = fig0.colorbar(im, ax=ax0.tolist(),extendrect=True,pad=0.05,aspect=12,shrink=.95)
        cbar.set_ticks(niveles[0::2])
        cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in niveles[0::2]])
        cbar.ax.set_title('$\omega / (L U_\infty)$ ');
        dirsave = self.dir_out + 'figs'
        if os.path.isdir(dirsave) == False:
                os.mkdir(dirsave)
        nomsave = dirsave+'/'+self.name+'_vortex_arrange.png'
        fig0.savefig(nomsave);
        if ver==0:
            plt.close(fig0);

        return
    def mean_flow_portrait(self,ver=0,maxvort=5,maxvx=1.2):
        fig0,ax0 = plt.subplots(1,1,sharex=True,sharey=True,figsize= (12,4));
        fig1,ax1 = plt.subplots(1,1,sharex=True,sharey=True,figsize= (12,4));
        fig2,ax2 = plt.subplots(1,1,sharex=True,sharey=True,figsize= (12,4));
        x,y,vort,vx,vy,q = [self.x,self.y,self.vort,self.vx,self.vy,self.q]
        vxm,vym = [vx.mean(0)/self.Uinf,vy.mean(0)/self.Uinf]
        Um = (vxm**2+vym**2)**.5
        niveles1 = np.linspace(-maxvort,maxvort,21)
        niveles1 = np.linspace(vort.mean(0).min()*0.8,vort.mean(0).max()*0.8,21)
        niveles2 = np.linspace(0,maxvx,21)
        niveles3=  np.linspace(0,q.mean(0)[x>1.5].max(),21)
        ax0i = ax0
        im1 = ax0i.contourf(x,y,vort.mean(0).T,levels=niveles1,extend='both')
        im2 = ax1.contourf(x,y,Um.T,levels=niveles2,extend='both')
        im1b = ax0i.contour(x[x>.5],y,q.mean(0)[x>.5].T,colors='k',levels=niveles3[::2],extend='both')
        im3 = ax2.contourf(x,y,q.mean(0).T,levels=niveles3,extend='both')
        ax0i.grid()
        ax1.grid()
        ax2.grid()
        ax1.set_ylim([-.6,.6])
        ax1.set_xlim([-.5,x.max()])
        ax1.set_xlabel('$x/L$')
        ax1.set_ylabel('$y/L$')  
        
        ax2.set_ylim([-.6,.6])
        ax2.set_xlim([-.5,x.max()])
        ax2.set_xlabel('$x/L$')
        ax2.set_ylabel('$y/L$') 
        
        ax0i.set_ylim([-.6,.6])
        ax0i.set_xlim([-.5,x.max()])
        ax0i.set_xlabel('$x/L$')
        ax0i.set_ylabel('$y/L$')
        fig0.tight_layout()
        fig1.tight_layout()
        fig2.tight_layout()
        #
        cbar = fig0.colorbar(im1, ax=ax0i,extendrect=True,pad=0.05,aspect=12,shrink=.95)
        cbar2 = fig1.colorbar(im2, ax=ax1,extendrect=True,pad=0.05,aspect=12,shrink=.95)
        cbar3 = fig2.colorbar(im3, ax=ax2,extendrect=True,pad=0.05,aspect=12,shrink=.95)
        cbar2.set_ticks(niveles2[0::2])
        cbar.set_ticks(niveles1[0::2])
        cbar3.set_ticks(niveles3[0::2])
        
        cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in niveles1[0::2]])
        cbar.ax.set_title('$\omega / (L U_\infty)$ ',fontsize=16);
        cbar2.ax.set_yticklabels(['{:.2f}'.format(x) for x in niveles2[0::2]])
        cbar2.ax.set_title('$U = (u_x^2+u_y^2)^{1/2}$ ',fontsize=16);
        
        cbar3.ax.set_yticklabels(['{:.3f}'.format(x) for x in niveles3[0::2]])
        cbar3.ax.set_title('$q$ ',fontsize=16);
     
        dirsave = self.dir_out + 'figs'
        if os.path.isdir(dirsave) == False:
                os.mkdir(dirsave)
        nomsave = dirsave+'/'+self.name+'_vortex_mean.png'
        nomsave2 = dirsave+'/'+self.name+'_velocity_mean.png'
        nomsave3 = dirsave+'/'+self.name+'_qmean.png'
        fig0.savefig(nomsave);
        fig1.savefig(nomsave2);
        fig2.savefig(nomsave3);
        if ver==0:
            plt.close(fig0);
            plt.close(fig1);
            plt.close(fig2)
        return
    
    def analisis_circ(self,plotear=0,errorp=10):
        B = np.load(self.dir_out+'circulacion/'+self.name+'.npz')
        positions_vortex = B['positions_vortex']
        positions_circ = B['positions_circ']

        hist_yv = np.histogram(np.abs(B['positions_circ'])[:,2],bins=10)
        Circ_m = hist_yv[1][hist_yv[0].argmax()]


        circs = positions_circ[:,2][np.abs((np.abs(positions_circ[:,2]) - Circ_m)) / Circ_m *100 < errorp ]
        yvs = positions_circ[:,1][np.abs((np.abs(positions_circ[:,2]) - Circ_m)) / Circ_m *100 < errorp ]
        xvs = positions_circ[:,0][np.abs((np.abs(positions_circ[:,2]) - Circ_m)) / Circ_m *100 < errorp ]
        self.yi_max = np.abs(yvs).max() - np.abs((yvs.max()+ yvs.min())/2)
        self.gamma_m = np.abs(circs).mean()
        self.yivs = positions_circ[:,1]
        self.xivs = positions_circ[:,0]
        if plotear == 1:
            yis = self.yivs
            xis = self.xivs
            xi1p = xis[yis>0]
            xi1n = xis[yis<0]
            yi1p = yis[yis>0]
            yi1n = yis[yis<0]
            fig5,ax5 = plt.subplots(1,1,figsize=(5,5))
            ax5.plot(xi1p,yi1p,'bs',markersize=9,markerfacecolor='none')
            ax5.plot(xi1n,yi1n,'ro',markersize=9,markerfacecolor='none')
            titulo = 'U%.2f f%.2f'%(self.Uinf,self.freq)
            ax5.set_title(titulo)
            fig5.savefig(self.name+'_vortex_coords.png')
            plt.close(fig5)
        return
    def Drag(self,x_integs,ylim=1.5,carga=0,graba=0):
        nameout = self.dir_out+'drag/'+self.name+'_drag.npz'
        #print(nameout)
        if np.logical_and(os.path.isfile(nameout)==True,carga==1)==True:
          print('carga '+nameout)
          A = np.load(nameout)
          self.Fza1,self.Fza2,self.Fza3,self.Fza4 = [A['Fza1'],A['Fza2'],
                                                     A['Fza3'],A['Fza4'],]
          self.x_integs = A['x_integs']
          self.y_integ = A['y_integ']
        else:  
          ylim=1.5
          x,y = [self.x,self.y]
          y_integ = y[np.abs(y)<ylim]
          vxm = self.vxm / self.Uinf
          vxstd = self.vxstd / self.Uinf
          vxvystd = self.vxvystd / self.Uinf
          vystd = self.vystd / self.Uinf

          fza1_l,fza2_l,fza3_l,fza4_l = [[],[],[],[]]
          for xi in x_integs:
              indice_x = np.abs(x-xi).argmin()
              #print(indice_x)
              vxm_integ = vxm[indice_x,np.abs(y)<ylim]
              vystd_integ = vystd[indice_x,np.abs(y)<ylim]
              vxstd_integ = vxstd[indice_x,np.abs(y)<ylim]
              vxvystd_integ = vxvystd[indice_x,np.abs(y)<ylim] 
              Fza1 = np.trapz(vxm_integ*(1-vxm_integ),y_integ) 

              Fza2 = Fza1  + np.trapz(vystd_integ**2,y_integ)

              Fza3 = Fza2 - np.trapz(vxstd_integ**2,y_integ) 
              Fza4 = Fza3 - np.trapz(vxvystd_integ**2,y_integ)
              fza1_l.append(Fza1)
              fza2_l.append(Fza2)
              fza3_l.append(Fza3)
              fza4_l.append(Fza4)
          
          self.Fza1,self.Fza2,self.Fza3,self.Fza4 = [np.asarray(fza1_l),np.asarray(fza2_l),
                                                    np.asarray(fza3_l),np.asarray(fza4_l)]
          self.x_integs = np.asarray(x_integs)
          self.y_integ = y_integ
          #print(graba)
          if graba==1:
            dictsal = {'Fza1':self.Fza2,'Fza2':self.Fza2,'Fza3':self.Fza3,'Fza4':self.Fza4,
                      'x_integs':self.x_integs,'y_integ':y_integ}
            np.savez(nameout,**dictsal)
        return 
    
    def Drag_Hall(self,L_hall):
        x,y = [self.x,self.y]
        u = self.vxm-1
        dx = np.diff(x).min()  
        aux1 = []
        xdoms = []
        for x0 in np.arange(1,x.max()-L_hall,dx):
            indice_x = np.logical_and(x>=x0,x<x0+L_hall)
            indice_y = np.logical_and(y>=-1.5,y<=1.5)
            x_dom = x[indice_x]
            y_dom = y[indice_y]
            aux1.append(-1.2*u[indice_x,:][:,indice_y].sum()*dx*dx)           
            xdoms.append(x_dom)
            #raise ValueError()
        #return
        self.Fza_h  = np.asarray(aux1)*self.freq*self.L/self.Uinf
        self.xdoms_h = np.asarray(xdoms)
        return 
    def vort_calc(self,rango):        
        self.vort_i,self.q_i,self.s_i =  np.tile(np.zeros_like(self.vx[rango,]),[3,1,1,1])
        dx = np.diff(self.x).min()
        vx = self.vx[rango,]
        for j,vy in enumerate(self.vy[rango,]):
            dvxdx,dvxdy = np.gradient(vx[j],dx)
            dvydx,dvydy = np.gradient(vy,dx)
            self.vort_i[j]= (dvydx-dvxdy)
            self.s_i[j]  = dvxdy + dvydx
            self.q_i[j] =  (dvydx-dvxdy)**2 - (dvxdy + dvydx)**2
        return    
        

    
