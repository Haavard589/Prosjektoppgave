# -*- coding: utf-8 -*-
from pyxem.utils import indexation_utils as iutls
from pyxem.utils import plotting_utils as putls
import pyxem as pxm #Electron diffraction tools based on hyperspy


from orix.quaternion import Orientation, symmetry
from orix.vector.vector3d import Vector3d
from orix import plot
from orix.crystal_map.crystal_map import CrystalMap


import numpy as np 

from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors #Some plotting color tools

from Utils.GetDiffLibrary import GetDiffLibrary, GetDiffLibrary_by_filepath

import hyperspy.api as hs #General hyperspy package
from pathlib import Path

from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_minimum
from skimage import measure
from tqdm import tqdm


class Template_matching:
    def __init__(self, signal = None, elements = None):
        self.signal = signal
        self.image = None
        self.orientations = None
        self.library = None
        self.template_indices = None
        self.angles = None
        self.correlations = None
        self.mirrored_angles = None 
        self.mirrored_correlations = None
        
        self.symmetries = None
        self.space_groups = None
        self.strucures = None
        
        self.orientations = None
        
        self.result = None  
        self.phasedict = None
        
        self.xmap = None 
        self.crystal_number = None
        
        self.cif_files = None
        self.file_location = None
        self.scatter = None
        
        
        self.elements = elements
        self.set_symmetry()
        
       
        
    def set_symmetry(self, elements = None, cif_file = None):  
        #Function used to initilize the symmetry of the mineral, 
        #such as space group and crystal family 
        
        if elements is not None:
            self.elements = elements
            
        self.symmetries   = [""]*len(self.elements)
        self.space_groups = [""]*len(self.elements)
        self.strucures    = [""]*len(self.elements)
        
        for i, element in enumerate(self.elements):
            if element == "muscovite" or element == "muscovite2":
                self.symmetries[i] = symmetry.C2h # C2h is monoclinic
                self.space_groups[i] = 15
                self.strucures[i] = "monoclinic"
    
            if element == "muscoviteT":
                self.symmetries[i] = symmetry.D3
                self.space_groups[i] = 151
                self.strucures[i] = "trigonal"
    
    
            if element == "illite":
                self.symmetries[i] = symmetry.C2h
                self.space_groups[i] = 12
                self.strucures[i] = "monoclinic"
                
            if element == "aluminium":
                self.symmetries[i] = symmetry.Oh
                self.space_groups[i] = 225
                self.strucures[i] = "cubic"
                
            if element == "LNMO":
                self.symmetries[i] = symmetry.C1
                self.space_groups[i] = 227
                self.strucures[i] = "cubic"
                
            if element == "CeAlO3 ":
                self.symmetries[i] = symmetry.S4
                self.space_groups[i] = 223
                self.strucures[i] = "tetragonal"    
        
        if cif_file is None:
            self.cif_files = [r"C:\Users\hfyhn\Documents\Skole\Fordypningsoppgave\Data\cif" + "\\" + element + ".cif" for element in self.elements]
        else:
            self.cif_files = cif_file
    
    def read_file(self, file, processing_type = None,lazy = True, area = 0, element = None):
        location = r"C:\Users\hfyhn\Documents\Skole\Fordypningsoppgave\Data\excite cambridge"
        if element is None:
            element = self.elements[0]
        if element == "muscovite":
            location += r"\D5 muscovite"
                
        if element == "illite":
            location += r"\A5 illite"
        
        if area == 0:
            if element == "muscovite":
                location += r"\20230711_124805"
            elif element == "illite":
                location += r"\20230710_134854"
                
        else:
            location += "\\" + area
        
        location += r"\hyperSpy"
        
        if isinstance(file, int) and -1 < int(file) < 10:
            location += r"\crystal_" + str(file)
            self.crystal_number = file
        else:
            location += "\\" + file
        
        if processing_type is None:
            location += ".hspy"
        else:
            location += "_" + processing_type + ".hspy"

        
        self.file_location = Path(location)
        
        self.signal = hs.load(location, lazy=lazy)
            
    def save_signal(self,name = None, add_on = False, location = None, overwrite = True):
        if location is None:
            if name is None:    
                output_directory = str(self.file_location) 
            else:  
                if add_on:
                    output_directory = str(self.file_location) + "_" + name + ".hspy"
                else:
                    output_directory = str(self.file_location.parent) + "\\" + name + ".hspy"

        else:
            output_directory = location + "\\" + name + ".hspy"
        try:
            self.signal.save(output_directory,overwrite = overwrite)
        except OSError:
            print("Failed to save file "+ output_directory)

        
    def correct_shifts_COM(self, com_mask, nav_mask=None, plot_results=False, inplace=False):
        com = self.signal.center_of_mass(mask=com_mask)
        if plot_results:
            com.get_bivariate_histogram().plot()
            
        beam_shift = pxm.signals.BeamShift(com.T)
        beam_shift.make_linear_plane(mask=nav_mask)

    
        x_shift, y_shift = [beam_shift.isig[ax] - self.signal.axes_manager.signal_shape[ax]/2.0 for ax in (0, 1)]
                
        dp_max_before = self.signal.max(axis=[0, 1])
        
        #A trick to make sure that the shifted signal contains the same metadata etc as the original signal. Might not be needed...
        if not inplace:
            shifted_signal = self.signal.deepcopy()
        else:
            shifted_signal = self.signal
        
        shifted_signal.shift_diffraction(x_shift, y_shift, inplace=True)
        
        dp_max_after = shifted_signal.max(axis=[0, 1])
            
        if plot_results:
            hs.plot.plot_images([dp_max_before, dp_max_after], overlay=True, colors=['w', 'r'], axes_decor='off', alphas=[1, 0.75])
        
        return shifted_signal, x_shift, y_shift
        
    
    def loop_COM(self, treshold = 2E5, nav_mask=None, plot_results=False, inplace=False):
        
        max_now = 0
        max_prev = treshold + 1
        max_signal = self.signal.max(axis = [0,1]).data
        prev_max_signal = self.signal.max(axis = [0,1]).data
        counter = 0
        while abs(max_prev - max_now) > treshold:
          
            s = self.signal.sum(axis = [0,1])
            m = np.argmax(s.data)
            shape = s.axes_manager.signal_shape
            x0, y0 = m % shape[0], m // shape[1]
            
            
            self.signal, x, y = self.correct_shifts_COM((x0, y0, 10), plot_results=False)
            max_prev = max_now
            prev_max_signal = max_signal
            max_signal = self.signal.max(axis = [0,1]).data
            max_now = np.sum(np.abs(max_signal - prev_max_signal))

            counter += 1 
            if counter > 7:
                break
        
        return x, y
    
    def update_axes_manager(self, step_size, dx, dy,cl, beam_energy):
        names = ['x', 'y', 'kx', 'ky']
        scan_units = "nm"
        
        offsets = [0, 0, -dx*self.signal.axes_manager[2].size/2, -dy*self.signal.axes_manager[2].size/2]

        scales = [step_size[0], step_size[1], dx, dy]
        units = [scan_units, scan_units, '$Å^{-1}$', '$Å^{-1}$']
        for ax_no, (name, scale, offset, unit) in enumerate(zip(names, scales, offsets, units)):
            self.signal.axes_manager[ax_no].name = name
            self.signal.axes_manager[ax_no].scale = scale
            self.signal.axes_manager[ax_no].units = unit    
            self.signal.axes_manager[ax_no].offset = offset

            
        self.signal.metadata.Acquisition_instrument.TEM.beam_energy = beam_energy
        self.signal.metadata.Acquisition_instrument.TEM.Detector.Diffraction.camera_length = cl
        
    def treshold(self, spot_value, background_value, with_mask = True, mask_folder = None):
        
        image = self.signal.data 
        image = np.where(image > 0.0, 1, 0.0)
        
        if mask_folder is None:
            mask_folder = self.file_location.parent
        mask = np.load(mask_folder)
        index = np.load(mask_folder + r"\mask_indices.npy").astype(int)
        mask = mask[int(index[self.crystal_number][0]):int(index[self.crystal_number][1]), int(index[self.crystal_number][2]):int(index[self.crystal_number][3])]
        
        self.signal.data = np.where(mask[:, :, np.newaxis, np.newaxis] == 0, 0, image)
        #plt.imshow(mask[int(index[3][0]):int(index[3][1]), int(index[3][2]):int(index[3][3])])
        #plt.imshow(image)

          
    def delete_vacuum(self):        
        
        mask_indices = np.load(str(self.file_location.parents[1]) + r"\mask_indices.npy")
        
        mask = np.load(str(self.file_location.parents[1]) + r"\mask.npy")
        
        
        index = mask_indices[self.crystal_number]
        plt.figure()
        plt.imshow(mask)
        plt.figure()
        plt.imshow(mask[int(index[0]):int(index[1]),int(index[2]):int(index[3])])
        mask = mask[int(index[0]):int(index[1]),int(index[2]):int(index[3])]
        plt.show()
        self.signal.data = np.where(mask[:, :, np.newaxis, np.newaxis] == 0, 0, self.signal.data)

    def generate_masks(self):
        
        img = self.signal.T.sum().data
        plt.imshow(img)
        img = np.abs(img)
        img = img / np.max(img)
    
        img = 1/(1 + np.exp(-10 * img))
        img = img / np.max(img)
        treshold = threshold_minimum(img)
        treshold_img = img > treshold
        img_labeled,island_count = measure.label(treshold_img,background=0,return_num=True,connectivity=1)
        indices = np.empty(shape = [island_count, 4])
        skipp = 0
        for i in range(island_count):
            
            island = np.where(img_labeled == i + 1)
            left, right = np.min(island[0]), np.max(island[0])+1
            top, buttom = np.min(island[1]), np.max(island[1])+1
            if (buttom - top)*(right - left) < 10:
                skipp += 1
                continue
            indices[i - skipp] = np.array([left, right, top, buttom], dtype=int)
    
    
        indices = np.delete(indices, range(island_count - skipp, island_count), axis = 0)
    
        with open(str(self.file_location.parents[1]) + "\\" + 'mask_indices.npy', 'wb') as f:
            np.save(f, indices)
            
        with open(str(self.file_location.parents[1]) + "\\" + 'mask.npy', 'wb') as f:
            np.save(f, treshold_img)


    def isolate_minerals(self):
            
        mask_indices = np.load(str(self.file_location.parents[1]) + r'\mask_indices.npy')
        
        #masked_data = [(data.T * mask).T.inav[int(m[2]):int(m[3]),int(m[0]):int(m[1])] for m in mask_indices]
        masked_data = [self.signal.inav[int(m[2]):int(m[3]),int(m[0]):int(m[1])] for m in mask_indices]

        
        for i, d in enumerate(masked_data):
            d.save(str(self.file_location.parents[1]) + r"\hyperSpy\crystal_"+str(i), overwrite = True)

        return 

    
    def find_blob(self, x, y, Gaussian_division = 64.0, DoG = (4.0,5.0), blur = 3.0, treshold = 2.0):

        data = self.signal.data[x,y].copy()
        
        Glauss_flat = gkern(sig = Gaussian_division)
        
        data = data / (Glauss_flat / np.max(Glauss_flat))
        
    
        gauss = gaussian_filter(data, sigma=DoG[0])
        data  -= gaussian_filter(gauss, sigma=DoG[1])
        #image = np.where(image < 2E4, image, 0.0)
    
        data = gaussian_filter(data, sigma=blur)
    

    
        scatters_y = [] 
        
        for j in range(256):
            point = find_extrema(data[:,j], treshold=treshold)
            scatters_y.append(point)  
            
      
        scatters_x = [] 
        for j in range(256):
            point = find_extrema(data[j,:], treshold=treshold)
            scatters_x.append(point)  
            
            
        scatters = []
        for i,row in enumerate(scatters_y):
            for element in row:
                if i in scatters_x[element]:
                    scatters.append((i,element))
    
        return scatters
    
    def store_blobs(self):
        
        
        mask = np.load(str(self.file_location.parents[1]) + r"\mask.npy")
        mask_indices = np.load(str(self.file_location.parents[1]) + r"\mask_indices.npy").astype(int)
        
        mask = mask[mask_indices[self.crystal_number,0]:mask_indices[self.crystal_number,1],mask_indices[self.crystal_number,2]:mask_indices[self.crystal_number,3]]
        plt.imshow(mask)
        
        
    
        all_blobs = np.zeros((mask_indices[self.crystal_number,1]-mask_indices[self.crystal_number,0], mask_indices[self.crystal_number,3]-mask_indices[self.crystal_number,2]), dtype=object)
    
        #all_blobs = np.zeros((len(signal.data), len(signal.data[1])), dtype=object)
        
        for x in tqdm(range(0, mask_indices[self.crystal_number,1] - mask_indices[self.crystal_number,0])):
            for y in range(0, mask_indices[self.crystal_number,3] - mask_indices[self.crystal_number,2]):
        #for x in tqdm(range(0, len(signal.data))):
        #    for y in range(0, len(signal.data[1])):    
                if not mask[x, y]:
                   all_blobs[x, y] = 0.0
                   continue
                
        
                centroids = self.find_blob_update(x,y)
                all_blobs[x, y] = centroids
        
        np.save(str(self.file_location.parents[1]) + r"\all_blobs_"+str(self.crystal_number)+".npy", all_blobs)

    
    def load_blobs(self, radius = 5):

        
        blobs = np.load(str(self.file_location.parents[1]) + r"\all_blobs_"+str(self.crystal_number)+".npy", allow_pickle=True)
        #print(np.max(blobs))
        with tqdm(total=(len(blobs)) * len(blobs[0]), position=0, leave=True) as pbar:
            for y, blob in tqdm(enumerate(blobs)):
                for x, b in tqdm(enumerate(blob)):
                    if isinstance(b, float):
                        self.signal.data[y,x] = 0.0
                        pbar.update()
                        continue
                    
                    mask = np.zeros((256,256))
                    for c in b:
                        mask += create_circular_mask(256, 256, center=c, radius=radius)
        
                    self.signal.data[y,x] = np.where(mask > 0.0, self.signal.data[y,x], 0.0)
           
                    pbar.update()
                



    def compute(self):
        self.signal.compute()
    
    def init_lib(self, filepath):
        self.library = GetDiffLibrary_by_filepath(filepath)

    def create_lib(self, resolution = 2.0, minimum_intensity=1E-20, max_excitation_error=1.7E-2, force_new = False, deny_new = False):
        diffraction_calibration = self.signal.axes_manager[-1].scale
        camera_length = self.signal.metadata.Acquisition_instrument.TEM.Detector.Diffraction.camera_length
        precession_angle = self.signal.metadata.Acquisition_instrument.TEM.rocking_angle


        self.library = GetDiffLibrary(diffraction_calibration = diffraction_calibration, 
                                camera_length = camera_length,
                                half_radius = np.min(self.signal.axes_manager.signal_shape)//2,
                                reciprocal_radius = np.max(np.abs(self.signal.axes_manager[-1].axis)),
                                resolution = resolution,
                                make_new=force_new,
                                grid = None, 
                                minimum_intensity=minimum_intensity,
                                max_excitation_error=max_excitation_error,
                                precession_angle = precession_angle,
                                cif_files = self.cif_files,
                                elements = self.elements,
                                structures = self.strucures, 
                                accelerating_voltage = self.signal.metadata.Acquisition_instrument.TEM.beam_energy,
                                deny_new_create = deny_new
                                )
        
    
    def test_lib_param(self, minimum_intensitys=[1E-20], max_excitation_errors=[1.7E-2]):
        #Tests if the parameters for the library create a library that looks okay
        diffraction_calibration = self.signal.axes_manager[-1].scale
        half_radius = np.min(self.signal.axes_manager.signal_shape)//2
        reciprocal_radius = np.max(np.abs(self.signal.axes_manager[-1].axis))
        print(half_radius, diffraction_calibration, reciprocal_radius)
        
        camera_length = self.signal.metadata.Acquisition_instrument.TEM.Detector.Diffraction.camera_length
        precession_angle = self.signal.metadata.Acquisition_instrument.TEM.rocking_angle
        

        for minimum_intensity in minimum_intensitys:
            for max_excitation_error in max_excitation_errors:
                self.library = GetDiffLibrary(diffraction_calibration = diffraction_calibration, 
                                        camera_length = camera_length,
                                        half_radius = half_radius,
                                        reciprocal_radius = reciprocal_radius,
                                        resolution = 16,
                                        make_new = True,
                                        grid = None, 
                                        minimum_intensity=minimum_intensity,
                                        max_excitation_error=max_excitation_error,
                                        precession_angle = precession_angle,
                                        cif_files = self.cif_files,
                                        elements = self.elements,
                                        structures = self.strucures, 
                                        accelerating_voltage = self.signal.metadata.Acquisition_instrument.TEM.beam_energy
                                        )
                template = np.argmin(np.array([np.linalg.norm(x - np.array([0,0,90])) for x in self.library[self.elements[0]]['orientations']]))

                self.scatterplot(template, str(minimum_intensity) + " | " + str(max_excitation_error))

   
    def scatterplot(self, title, template = -1, euler_angle = None):
        plt.figure()
        

        
        for element in self.elements:
            if template == -1 and not euler_angle is None:
                template = np.argmin(np.array([np.linalg.norm(x - euler_angle) for x in self.library[self.elements]['orientations']]))

            
            coords = self.library[element]["pixel_coords"][template]
            plt.scatter(coords[:,0], coords[:,1], marker = "x", label = element)
            
            plt.legend(loc='upper right')
            plt.axis('equal')
            plt.title("Euler angle " + str(self.library[element]["orientations"][template]) + " | " + title)

        
    def entire_dataset_single_signal(self, x = -1, y = -1, element = "muscovite", plot = True, image = None, nbest = 5, intensity_transform_function = lambda x:x**0.15, template = -1, euler_angle = -1):
        frac_keep = 1  # if frac_keep < 1 or 1 < n_keep < number of templates then indexation will be performed on the
        n_keep = None
        if image is None: 
            s = self.signal.inav[x,y]
       
            try:
                s.compute()
            except Exception:
                pass
            self.image = s.data
        else:
            self.image = image

        self.orientations = None
        self.template_indices = None
        self.angles = None
        self.correlations = None
        self.mirrored_angles = None
        self.mirrored_correlations = None
        
       
        self.template_indices, self.angles, self.correlations, self.mirrored_angles\
            , self.mirrored_correlations = iutls.correlate_library_to_pattern(
            self.image, 
            self.library[element]['simulations'], 
            frac_keep=frac_keep, 
            n_keep=n_keep, 
            delta_r = 0.3, 
            delta_theta = 1,
            max_r = None,
            intensity_transform_function= intensity_transform_function, #lambda x: np.where(x != 0.0, x**0.4, -0.4),#
            find_direct_beam = False, 
            direct_beam_position = (128,128),
            normalize_image = True, 
            normalize_templates = True,
        )
    
        self.orientations = Orientation.from_euler(self.library[element]['orientations'][self.template_indices], symmetry=self.symmetries[0], degrees=True)
        argmax = np.argmax(self.correlations)
        if not plot:
            return argmax, self.angles[argmax]
        
        self.plot_simulation(element = element, nbest = nbest, template = template, euler_angle = euler_angle)
        return argmax, self.angles[argmax]
    def IPF_map(self):
        for i, element in enumerate(self.elements):
            fig = plt.figure(figsize=(8, 8))
            
            max_correlation = np.max(np.stack((self.correlations[i], self.mirrored_correlations[i])))
            max_angle = np.max(np.stack((self.angles, self.mirrored_angles)))
                    
            
            ax0 = fig.add_subplot(221, projection="ipf", symmetry=self.symmetries[i])
            ax0.scatter(self.orientations[i], c=self.correlations[i]/max_correlation, cmap='inferno')
            ax0.set_title('Correlation for element '+ element)
            
            ax1 = fig.add_subplot(222, projection="ipf", symmetry=self.symmetries[i])
            ax1.scatter(self.orientations[i], c=self.mirrored_correlations[i]/max_correlation, cmap='inferno')
            ax1.set_title('Correlation (m) for element '+ element)
            
            ax2 = fig.add_subplot(223, projection="ipf", symmetry=self.symmetries[i])
            ax2.scatter(self.orientations[i], c=self.angles[i]/max_angle, cmap='inferno')
            ax2.set_title('Angle for element '+ element)
            
            ax3 = fig.add_subplot(224, projection="ipf", symmetry=self.symmetries[i])
            ax3.scatter(self.orientations[i], c=self.mirrored_angles[i]/max_angle, cmap='inferno')
            ax3.set_title('Angle (m) for element '+ element)
            
            plt.colorbar(ScalarMappable(norm=mcolors.Normalize(0, max_angle), cmap='inferno'), ax=ax2)
            plt.colorbar(ScalarMappable(norm=mcolors.Normalize(0, max_angle), cmap='inferno'), ax=ax3)
        
    def IPF_maping(self, nbest, orientation = None, image = None, scatter = None):
        correlations_list = np.argsort(self.correlations)[-nbest:]
        mirrored_correlations_list = np.argsort(self.mirrored_correlations)[-nbest:]
        orientations = self.orientations[correlations_list]
        #orientations_m = self.orientations[mirrored_correlations_list]

        correlations = self.correlations[correlations_list] - np.min(self.correlations[correlations_list])
        
        mirrored_correlations = self.mirrored_correlations[mirrored_correlations_list]

        fig = plt.figure(figsize=(10, 10), constrained_layout = True)
       

        
        plt.rcParams['font.size'] = 26
        max_correlation = np.max(np.stack((correlations, mirrored_correlations)))
        
        if orientation is None: 
            ax0 = fig.add_subplot(111, projection="ipf", symmetry=self.symmetries[0])
            
        else:
            if image is None:
                    
                ax0 = fig.add_subplot(121, projection="ipf", symmetry=self.symmetries[0])
                ax1 = fig.add_subplot(122, projection="ipf", symmetry=self.symmetries[0])
                ax1.set_title('Correlation')
                ax1.scatter(Orientation.from_euler([orientation], symmetry=self.symmetries[0], degrees=True), c=np.linspace(0,1,num=len(orientation)), cmap='coolwarm')
    
                ax1.set_title('Correlation')
            else:
               ax0 = fig.add_subplot(221, projection="ipf", symmetry=self.symmetries[0])
               ax1 = fig.add_subplot(222, projection="ipf", symmetry=self.symmetries[0])
               o = Orientation.from_euler(orientation, symmetry=self.symmetries[0], degrees=True)
               #o[0].
               #ax1.set_title("Misorientation " + str(o[0].angle_with(o[1], degrees=True)[0]))
               ax1.set_title("2)")

               ax1.scatter(o[0], c = "blue",s = 100, label = "Background")
               ax1.scatter(o[1], c = "red", s = 100, label = "Best fit")
               
               handles, labels = ax1.get_legend_handles_labels()
               ax1.legend(handles[-2:], labels[-2:], loc='upper right',bbox_to_anchor=(1, -0.15), ncol=1)
               #ax1.legend(["Background", "Template match"], loc = "upper right")
               
               ax2 = fig.add_subplot(223)
               #ax2.set_title('Euler angle \n Background' + str(orientation[0]) + "\n Template match" + str(orientation[1]))
               ax2.set_title('3)')
               
               ax2.imshow(image, cmap="Greys_r", norm ="symlog")
               ax2.scatter(scatter[:,0], scatter[:,1], marker = "x", color = "red")
               ax2.tick_params(left = False, right = False , labelleft = False , 
                       labelbottom = False, bottom = False) 
               ax3 = fig.add_subplot(224)
               ax3.set_title('4)')
               ax3.tick_params(left = False, right = False , labelleft = False , 
                       labelbottom = False, bottom = False) 
               
               ax3.spines['top'].set_visible(False)
               ax3.spines['right'].set_visible(False)
               ax3.spines['bottom'].set_visible(False)
               ax3.spines['left'].set_visible(False)
               oris = Orientation.stack(Orientation.from_euler(orientation, degrees=True)).squeeze()
               oris.symmetry = orientations[0].symmetry    
               oris.scatter(ec="k", s=100, c=[[0,0,1],[1,0,0]], figure = fig, position = (2,2,4))
               
               
        ax0.set_title('1)')
        ax0.scatter(orientations, c=correlations/max_correlation, cmap='inferno')
        
    def IPF_mapping_given_orientation(self, orientations, correlations, invert = False):
    
        fig = plt.figure(figsize=(8, 8))

        max_correlation = np.max(correlations)
    

        ax0 = fig.add_subplot(111, projection="ipf", symmetry=self.symmetries[0])
        #ax0.set_title('Correlation')

        print(len(orientations))
        orientations = Orientation.from_euler(orientations, symmetry=self.symmetries[0], degrees=True)
        ax0.scatter(orientations, c=invert - (correlations/max_correlation), cmap='coolwarm')
        
    def fund_plot(self, orientations, rgb_list):
        oris = Orientation.stack(orientations).squeeze()
        oris.symmetry = orientations[0].symmetry    
        oris.scatter(ec="k", s=100)
   
    def order_best(self, nbest = 5, find_mirror = True):
        best_templates = [0]*nbest
        best_angles = [0]*nbest
        best_correlation = [0]*nbest
        mirrored = [0]*nbest
        correlation_list = np.argsort(self.correlations)[-nbest:]
        mirrored_correlations_list = np.argsort(self.mirrored_correlations)[-nbest:]
        counter = 1
        m_counter = 1
        for i in range(nbest):
            
            ci = correlation_list[-counter]
            
            mci = mirrored_correlations_list[-m_counter]
            #print(ci,mci, counter, m_counter)
            if not find_mirror or self.correlations[ci] > self.mirrored_correlations[mci]:
                best_angles[i] = self.angles[ci]
                best_templates[i] = self.template_indices[ci]
                best_correlation[i] = self.correlations[ci]
                mirrored[i] = False
                counter += 1 
            
            else:
                best_angles[i] = self.mirrored_angles[mci]
                best_templates[i] = self.template_indices[mci]
                best_correlation[i] = self.mirrored_correlations[mci]
                mirrored[i] = True
                m_counter += 1 
                
        return best_templates, best_angles, best_correlation, mirrored
        
                   
    def plot_simulation(self, nbest = 1, element = "muscovite", template = -1, euler_angle = -1):
        plt.axis('off')
        best_templates, best_angles, best_correlation, mirrored = self.order_best(nbest, False)
    
        if template == -1 and euler_angle != -1:
            template = np.argmin(np.array([np.linalg.norm(x - euler_angle) for x in self.library[element]['orientations']]))

        if template != -1:
            putls.plot_template_over_pattern(self.image,
                                             self.library[element]['simulations'][template],
                                             in_plane_angle=self.angles[template],
                                             coordinate_system = "cartesian", 
                                             size_factor = 1,
                                             max_r = 200,
                                             mirrored_template=False,
                                             find_direct_beam=False,
                                             marker_color='r',
                                             cmap = "Blues",
                                             norm='symlog'
                                            )
            print("Actual pattern:", self.correlations[template])
        for i, template in enumerate(best_templates):
            plt.axis('off')
            print("\t\n",str(self.library[element]['orientations'][template] + np.array([best_angles[i],0,0])), best_correlation[i])
            
            putls.plot_template_over_pattern(self.image,
                                             self.library[element]['simulations'][template],
                                             in_plane_angle=best_angles[i],
                                             coordinate_system = "cartesian", 
                                             size_factor = 1,
                                             max_r = 200,
                                             mirrored_template=False,
                                             find_direct_beam=False,
                                             marker_color='r',
                                             cmap = "viridis",
                                             norm='symlog'
                                            )
            plt.axis('off')
            
    def template_match(self, intensity_transform_function = lambda x:x**0.4):
        frac_keep = 1

        self.result, self.phasedict = iutls.index_dataset_with_template_rotation(self.signal,
                                                                        self.library,
                                                                        phases = self.elements,
                                                                        n_best = 1,
                                                                        frac_keep = frac_keep,
                                                                        n_keep = None,
                                                                        delta_r = 1,
                                                                        delta_theta = 1,
                                                                        max_r = np.min(self.signal.axes_manager.signal_shape)//2,
                                                                        intensity_transform_function = intensity_transform_function,
                                                                        normalize_images = True,
                                                                        normalize_templates = True,
                                                                        )
        
        self.phasedict[len(self.elements)] = 'vacuum'
        self.result['phase_index'][np.isnan(self.result['correlation'])] = len(self.elements)
        print("Mean correlation", np.nanmean(self.result["correlation"]))
        
    def plot_template_matching(self, element = "muscovite"):
        if self.result is None:
            print("You must run templete match before you can plot!")
            return 
        px, py = [self.signal.axes_manager[ax].index for ax in (0, 1)] #Get the x-y coordinates to check
        n_sol = 0 # Select which solution to plot, should be an integer between 0 and `n_best-1`
        
        solution = self.result["orientation"] #Get the orientations of the  result
        correlations = np.nan_to_num(self.result["correlation"][:, :, n_sol].ravel()) #Get the correlation for
        
        #Plot IPF maps of the results
        orientations = Orientation.from_euler(solution, symmetry=self.symmetry, degrees=True)
        sols = orientations.shape[-1]
        fig = plt.figure(figsize=(sols*1.2, 2*1.2))
        max_correlation = np.max(np.nan_to_num(self.result["correlation"]))
        for n in range(sols):
            ax = fig.add_subplot(1, sols, n+1, projection='ipf', symmetry=self.symmetry)
            ax.scatter(orientations[:, :, n], c=correlations/max_correlation, cmap='inferno')
        
            
        # Get the results from the selected scan pixel and solution
        sim_sol_index = self.result["template_index"][py, px, n_sol]
        mirrored_sol = self.result["mirrored_template"][py, px, n_sol]
        in_plane_angle = self.result["orientation"][py, px, n_sol, 0] #! NOTE: the first angle should be the in plane angle! But the template in the resulting figure does not look correct - must check!
        # Get the template for the selected solution and pixel
        sim_sol = self.library[element]['simulations'][sim_sol_index]
        
        #Plot an IPF map and the diffraction pattern with template overlay in a single figure
        fig = plt.figure(figsize=(8, 4))
        
        ax0 = fig.add_subplot(121, projection="ipf", symmetry=self.symmetry)
        ax0.scatter(orientations[:, :, n_sol], c=correlations/np.max(correlations), cmap='inferno')
        ax0.scatter(orientations[py, px], c=np.arange(sols), cmap='Greys')
        ax0.set_title('Correlation')
        
        ax1 = fig.add_subplot(122)
        
        # plotting the diffraction pattern and template
        putls.plot_template_over_pattern(self.signal.get_current_signal().data,
                                         sim_sol,
                                         ax=ax1,
                                         in_plane_angle=in_plane_angle,
                                         coordinate_system = "cartesian", 
                                         size_factor = 10,
                                         norm=mcolors.SymLogNorm(0.03),
                                         max_r = 200,
                                         mirrored_template=mirrored_sol,
                                         find_direct_beam=True,
                                         cmap = "inferno",
                                         marker_color = "green"
                                        )
        for i in [ax0, ax1]:
            i.axis("off")
            
            
    
    def init_xmap(self):
        self.xmap = iutls.results_dict_to_crystal_map(self.result, self.phasedict, diffraction_library=self.library, index=0)
        for i, space_group in enumerate(self.space_groups):
            if np.size(self.xmap[self.elements[i]]) == 0:
                continue 
            self.xmap.phases[i].space_group = space_group
    
        self.set_xmap_step_size()
        
            
    def set_xmap_step_size(self):
        """Change the step size of an orix CrystalMap
        
        """
        x = self.xmap.x * self.signal.axes_manager[0].scale
        y = self.xmap.y * self.signal.axes_manager[1].scale
        rot = self.xmap.rotations
        phaseid = self.xmap.phase_id
        prop = self.xmap.prop
        is_in_data = self.xmap.is_in_data
        phaselist = self.xmap.phases
        new_xmap = CrystalMap(rotations = rot,
                            phase_id = phaseid,
                            x = x,
                            y = y,
                            prop = prop,
                            scan_unit = "nm",
                            is_in_data = is_in_data,
                            phase_list=phaselist)
        self.xmap = new_xmap
        
    def plot_orientations_mapping(self, no_correlation = True, correlation = True):
        vectors = [Vector3d.xvector(),
                   Vector3d.yvector(),
                   Vector3d.zvector()]

       
        if self.xmap is None:
            self.init_xmap()

        nx, ny = self.xmap.shape
        aspect_ratio = nx/ny
        figure_width = 4
        
        for i, element in enumerate(self.elements):
            if np.size(self.xmap[element]) == 0:
                continue 
            ckey = plot.IPFColorKeyTSL(self.symmetries[i])# Plot the key once
            fig = ckey.plot(return_figure=True)
            fig.set_size_inches(2, 2)
                
            
            for j, ploting in enumerate([no_correlation, correlation]):
                if not ploting:
                    continue

                for v in vectors:
                    ckey = plot.IPFColorKeyTSL(self.symmetries[i], direction=v)
                    overlay = None
                    if j == 1:
                        overlay = "correlation"
                        
                    if np.size(self.xmap[element]) == 0:
                        continue 
                    
                    
                    

                    
                    fig = self.xmap[element].plot(value = ckey.orientation2color(self.xmap[element].orientations), 
                                                  overlay = overlay, 
                                                  figure_kwargs={'figsize': (figure_width, figure_width*aspect_ratio)}, 
                                                  return_figure=True)
                    ax = fig.get_axes()[0]
                    ax.axis('off')

       
    def plot_correlation(self, update = False):
       if update or self.xmap is None:
           self.init_xmap()
           
       self.xmap.plot('correlation', colorbar=True)
        
       
    def plot_phase_map(self, update = False):
        if update or self.xmap is None:
            self.init_xmap()
            
        self.xmap.plot()
        
        
    def plot_orientations_overlay_correlation(self, element = "muscovite"):
        if self.result is None:
            print("You must run templete match before you can plot!")
            return 
        
        if self.xmap is None:
            self.xmap = iutls.results_dict_to_crystal_map(self.result, self.phasedict, diffraction_library=self.library, index=0)
            for i, space_group in enumerate(self.space_groups):
                self.xmap.phases[i].space_group = space_group
        
        vectors = [Vector3d.xvector(),
                   Vector3d.yvector(),
                   Vector3d.zvector()]
        
        ckey = plot.IPFColorKeyTSL(self.symmetry)# Plot the key once
        fig = ckey.plot(return_figure=True)
        fig.set_size_inches(2, 2)
        
        nx, ny = self.xmap.shape
        aspect_ratio = nx/ny
        figure_width = 4
        
        for v in vectors:
            ckey = plot.IPFColorKeyTSL(self.symmetry, direction=v)
            fig = self.xmap[element].plot(ckey.orientation2color(self.xmap[element].orientations), overlay="correlation", figure_kwargs={'figsize': (figure_width, figure_width*aspect_ratio)}, return_figure=True)
            ax = fig.get_axes()[0]
            ax.axis('off')
            
def find_extrema(data, treshold = 30E-5):
     point = [extrema(data[i:i+5]) for i in range(250)]
     
     point = [0] + [i+2 for i, x in enumerate(point) if x]
     if len(point) < 1:
         return []
     
     derivative = [greatest_derivative(data[point[i]:point[i+ 1]]) for i in range(len(point) - 1)]
     point = [point[i + 1] for i, x in enumerate(derivative) if x > treshold]

     return np.array(point)
 
def extrema(array):
    n = len(array)
    v = array[n//2]
    s1 = sum([(v - array[i]) for i in range(0,n//2)])
    s2 = sum([(v - array[i]) for i in range(n//2 + 1, n)])
    return (s1 > 0.0 and s2 > 0.0) #all(i >= v for i in array)

def greatest_derivative(array):
    if len(array) <  2:
        return False
    return max([array[i + 1] - array[i] for i in range(len(array) - 1)])

def gkern(l=256, sig=64, x = 0, y = 0):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss_x = np.exp(-0.5 * np.square(ax - x) / np.square(sig))
    gauss_y = np.exp(-0.5 * np.square(ax - y) / np.square(sig))

    kernel = np.outer(gauss_y, gauss_x)
    return kernel / np.sum(kernel)
 
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

