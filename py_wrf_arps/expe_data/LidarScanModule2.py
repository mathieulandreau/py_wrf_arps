import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpld
import os
import datetime as dt
import pytz
from time import time
#from suntime import Sun
from random import randrange
import seaborn
from statistics import mean
from scipy.stats import linregress

import matplotlib.cm as cm
import matplotlib as mpl
from warnings import warn
from scipy.signal import find_peaks
from scipy.special import gamma
from math import exp
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

NaN = float('NaN')
pd.options.display.float_format = "{:,.2f}".format

import netCDF4 as nc


class LidarScan2:
    """
    internal attributes :
        filename
        heights
        
    file architecture :
        wind_data
            time
            heights
            data_availability
            wind_speed
            mean_r2 (optional)
            mean_nrmse (optional)
            reliable_profiles (boolean)
        shear_data
            shear_time
            shear_exponent
            shear_exponent_r2
            shear_across_rotor
            shear_local_alpha
        
        llj_data
            llj_time
            llj_speed
            llj_direction
            llj_turning
            llj_height
            llj_strength
            llj_fall_off  
    """
    
    def __init__(self, ncfile, era=True, stability=True, st=None, en=None):
        
        
        self.filename = ncfile
        
        ff = nc.Dataset(ncfile, 'r', format='NETCDF4')
                
        self.heights = ff.groups['wind_data'].variables['height'][:].data
        height_names = [f'H{i+1}' for i in range(len(self.heights))]
        
        wind_time = ff.groups['wind_data'].variables['time']
        time_ref = wind_time.units
        wind_time = nc.num2date(wind_time[:], time_ref, only_use_cftime_datetimes=False, only_use_python_datetimes=True).data
        #wind_time = nc.num2date(wind_time[:], time_ref).data
        
        if st == None: st = wind_time[0];
        if en == None: en = wind_time[-1];
        
        try:
            data_availability = ff.groups['wind_data'].variables['data_availability'][:][:]
            self.data_availability = pd.DataFrame(data_availability, index=wind_time, columns=height_names)[st:en]
        except IndexError:
            print('Probably invalid start and/or end times')
        
        av_thr = 5 #availability threshold
        
        wind_speed = ff.groups['wind_data'].variables['wind_speed'][:][:]
        self.wind_speed = pd.DataFrame(wind_speed, index=wind_time,
                                       columns=height_names).where(self.data_availability >= av_thr)[st:en]
        
        wind_direction = ff.groups['wind_data'].variables['wind_direction'][:][:]
        self.wind_direction = pd.DataFrame(wind_direction, index=wind_time,
                                           columns=height_names).where(self.data_availability >= av_thr)[st:en]
        
        try:
            mean_r2 = ff.groups['wind_data'].variables['mean_r2'][:][:]
            self.mean_r2 = pd.DataFrame(mean_r2, index=wind_time, columns=height_names)[st:en]

            mean_nrmse = ff.groups['wind_data'].variables['mean_nrmse'][:][:]
            self.mean_nrmse = pd.DataFrame(mean_nrmse, index=wind_time, columns=height_names)[st:en]
        except:
            pass
#             print('oops')

#         print(ff.groups['wind_data'].variables.keys())

        reliable_profile = ff.groups['wind_data'].variables['reliable_profiles'][:].data.astype('bool')
        self.reliable_profile = pd.Series(reliable_profile, index=wind_time)[st:en]

        shear_time = ff.groups['shear_data'].variables['shear_time'][:]
        shear_time = nc.num2date(shear_time, time_ref, only_use_cftime_datetimes=False, only_use_python_datetimes=True).data
        
        shear_e = ff.groups['shear_data'].variables['shear_exponent'][:]
        shear_er2 = ff.groups['shear_data'].variables['shear_exponent_r2'][:]
        shear_across_rotor = ff.groups['shear_data'].variables['shear_across_rotor'][:]
        shear_local_e = ff.groups['shear_data'].variables['shear_local_alpha'][:]
        
        self.shear = pd.DataFrame({'exp': shear_e, 'r2': shear_er2, 'rotor': shear_across_rotor, 'local_exp': shear_local_e},
                                  index=shear_time)[st:en]
        
        llj_time = ff.groups['llj_data'].variables['llj_time'][:]
        llj_time = nc.num2date(llj_time, time_ref, only_use_cftime_datetimes=False, only_use_python_datetimes=True).data
        
        llj_speed = ff.groups['llj_data'].variables['llj_core_speed'][:]
        llj_direction = ff.groups['llj_data'].variables['llj_core_direction'][:]
        llj_turning = ff.groups['llj_data'].variables['llj_core_turning'][:]
        llj_height = ff.groups['llj_data'].variables['llj_core_height'][:]
        llj_strength = ff.groups['llj_data'].variables['llj_strength'][:]
        llj_fall_off = ff.groups['llj_data'].variables['llj_fall_off'][:]
        
        self.llj = pd.DataFrame({'speed': llj_speed, 'height': llj_height, 'direction': llj_direction, 'turning': llj_turning,
                                'strength': llj_strength, 'fall_off': llj_fall_off},
                                index=llj_time).astype({'height': 'int'})[st:en]
        
        def sun_exposure(timestamp, timezone='CET', location=(47.2943, -2.5119)):
            tz = pytz.timezone(timezone)
            lat, lon = location
            sun = Sun(lat,lon)
            sunrise = sun.get_sunrise_time(timestamp.date()).astimezone(tz)
            sunset = sun.get_sunset_time(timestamp.date()).astimezone(tz)
            if (tz.localize(timestamp.to_pydatetime()) >= sunrise) and (tz.localize(timestamp.to_pydatetime()) <= sunset):
                return True
            else: return False;
                
            
        #self.daylight = self.wind_speed.index.to_series().apply(sun_exposure)[st:en]

        
        ff.close()
        
        if era:
            erafile = r'data/ERA5/ERA5_feb_aug_v3.nc'
            
            f = nc.Dataset(erafile)
            raw_time = f.variables['time'][:]
            time = nc.num2date(raw_time, f.variables['time'].units, only_use_python_datetimes=True, only_use_cftime_datetimes=False)
            time = pd.to_datetime(time)
            
#             chp_lat = np.float32(47.3)
#             chp_lon = np.float32(-2.5)
            
            chp_lat = np.float32(47.25)
            chp_lon = np.float32(-2.5)
            
            sea_lat = np.float32(47.25)
            sea_lon = np.float32(-2.75)
            
            lons = f.variables['longitude'][:]
            lats = f.variables['latitude'][:]

            chp_lon_ind = list(lons).index(chp_lon)
            chp_lat_ind = list(lats).index(chp_lat)
            
            sea_lon_ind = list(lons).index(sea_lon)
            sea_lat_ind = list(lats).index(sea_lat)

            temperature = f.variables['t2m'][:, chp_lat_ind, chp_lon_ind]
            precipitation = f.variables['tp'][:, chp_lat_ind, chp_lon_ind]
            pressure = f.variables['msl'][:, chp_lat_ind, chp_lon_ind]
            cloud_cover = f.variables['tcc'][:, chp_lat_ind, chp_lon_ind]
            BL_height = f.variables['blh'][:, chp_lat_ind, chp_lon_ind]
            sst = f.variables['sst'][:, sea_lat_ind, sea_lon_ind]
            delta_t = f.variables['t2m'][:, sea_lat_ind, sea_lon_ind] - f.variables['sst'][:, sea_lat_ind, sea_lon_ind]
            u10 = f.variables['u10'][:, chp_lat_ind, chp_lon_ind]
            v10 = f.variables['v10'][:, chp_lat_ind, chp_lon_ind]
            u100 = f.variables['u100'][:, chp_lat_ind, chp_lon_ind]
            v100 = f.variables['v100'][:, chp_lat_ind, chp_lon_ind]

            ERA = pd.DataFrame({'precip': precipitation, 'temp_2m': temperature, 'pressure': pressure,
                               'cloud_cover': cloud_cover, 'bl_height': BL_height, 'delta_t': delta_t,
                               'sst': sst, 'u10': u10, 'v10': v10, 'u100': u100, 'v100': v100}, index=time)
            
            ERA['WS10'] = (ERA['u10']**2 + ERA['v10']**2)**0.5
            ERA['WS100'] = (ERA['u100']**2 + ERA['v100']**2)**0.5
            
            ERA['time_of_day'] = ERA.index.to_series().apply(lambda date: date.hour + date.minute/60)
            ERA['current_day'] = ERA.index.to_series().apply(lambda date: date.day)
            self.ERA = ERA.loc[self.wind_speed.index.min():self.wind_speed.index.max()]
            
            
            

        
    def __add__(self, other):
        
        if not ((len(self.heights) == len(other.heights)) and (sum(self.heights == other.heights) == len(self.heights))):
            print('Warning: heights are not the same')
            return None
            
        else:
            result = self
            result.wind_speed = pd.concat([result.wind_speed, other.wind_speed])
            result.wind_direction = pd.concat([result.wind_direction, other.wind_direction])
            result.shear = pd.concat([result.shear, other.shear])
            result.llj = pd.concat([result.llj, other.llj])
            result.data_availability = pd.concat([result.data_availability, other.data_availability])
            result.daylight = pd.concat([result.daylight, other.daylight])
            return result

    def statistics(self):
        
        chosen_heights_names = ['H1', 'H4', 'H7', 'H11', 'H15', 'H19', 'H21', 'H24', 'H27']
        chosen_heights = pd.DataFrame(self.heights, index=self.wind_speed.columns)
        chosen_heights = chosen_heights.loc[chosen_heights_names]
        height_line = ' \t'.join([f'{int(h)}' for h in np.array(chosen_heights)])
        mean_speed_line = '\t'.join([f'{ws:.2f}' for ws in np.array(self.wind_speed.mean().loc[chosen_heights_names])])
        max_speed_line = '\t'.join([f'{ws:.2f}' for ws in np.array(self.wind_speed.max().loc[chosen_heights_names])])
        
        av_cut = self.data_availability[chosen_heights_names].where(self.data_availability>=5)
        availability_line = '\t'.join([f'{av_cut[h].dropna().shape[0]*100/av_cut.shape[0]:.1f}%' for h in chosen_heights_names])
        
        print(f'File name: {self.filename}')
        print(f'Time: {self.wind_speed.index[0]} to {self.wind_speed.index[-1]}\n')
        print(f'Heights [m]: \t \t \t {height_line}')
        print(f'Max wind speeds [m/s]: \t \t {max_speed_line}')
        print(f'Mean wind speeds [m/s]: \t {mean_speed_line}')
        print(f'Share of reliable data: \t {availability_line}\n')
        
        print(f'Share of reliable profiles: {sum(self.reliable_profile)*100/len(self.wind_speed.index):.1f}%')
        print(f'LLJ occurrence: {len(self.llj.index)/sum(self.reliable_profile)*100:.1f}% ({len(self.llj.index)} cases)')
        
        
    def get_statistics(self):
        
        stats = pd.Series()
        
        stats['Reliable profiles [%]'] = f'{sum(self.reliable_profile)*100/len(self.wind_speed.index):.1f}'
        stats['Maximum wind speed [m/s]'] = f'{self.wind_speed.H6.max():.1f}'
        stats['Mean wind speed [m/s]'] = f'{self.wind_speed.H6.mean():.1f}'
        stats['Maximum shear exponent'] = f'{self.shear.local_exp.max():.2f}'
        stats['Minimum shear exponent'] = f'{self.shear.local_exp.min():.2f}'
        stats['Mean shear exponent'] = f'{self.shear.local_exp.mean():.2f}'
        stats['LLJ occurrence [%]'] = f'{len(self.llj.index)/sum(self.reliable_profile)*100:.1f}'
        stats['LLJ cases'] = f'{len(self.llj.index):.0f}'
        stats['Mean LLJ core speed [m/s]'] = f'{self.llj.speed.mean():.1f}'
        stats['Mean LLJ core height [m]'] = f'{self.llj.height.mean():.1f}'
        
        pie_loc = 'C'
        try:
            unstable = (self.zeta[pie_loc] < -0.2).sum()*100/len(self.zeta[pie_loc])
            stable = (self.zeta[pie_loc] > 0.2).sum()*100/len(self.zeta[pie_loc])
            neutral = ((self.zeta[pie_loc] >= -0.2) & (self.zeta[pie_loc] <= 0.2)).sum()*100/len(self.zeta[pie_loc])
            stats['Atmospheric stability: stable [%]'] = f'{stable:.1f}'
            stats['Atmospheric stability: unstable [%]'] = f'{unstable:.1f}'
            stats['Atmospheric stability: neutral [%]'] = f'{neutral:.1f}'
        except:
            pass
        
        return stats
    
    
        
    def windrose(self, height='H1', mode='wind_speed', ax=None, title=None):

        # this method is adapted from https://gist.github.com/phobson/41b41bdd157a2bcf6e14

        if mode not in ['wind_speed', 'llj', 'shear']:
            raise ValueError('Invalid mode')
        
        wr_data = pd.DataFrame()
        if mode == 'llj':
            wr_data['WindVel'] = self.llj.speed
            wr_data['WindDir'] = self.llj.direction
            if title==None: title = 'LLJ windrose';
            spd_bins = [0, 5, 10, 15, np.inf]
            units = 'm/s'
            
#             wr_data['WindVel'] = self.wind_speed.loc[self.llj.index][height]
#             wr_data['WindDir'] = self.wind_direction.loc[self.llj.index][height]
        elif mode == 'wind_speed':                
            wr_data['WindVel'] = self.wind_speed[height]
            wr_data['WindDir'] = self.wind_direction[height]
            if title==None: title = 'Windrose at hub height';
            spd_bins = [0, 5, 10, 15, np.inf]
            units = 'm/s'
            
        elif mode == 'shear':
            wr_data['WindVel'] = self.shear.local_exp
            wr_data['WindDir'] = self.wind_direction.H6.loc[self.shear.index]
            if title==None: title = 'Shear exponent windrose';
            spd_bins = [-np.inf, 0, 0.2, 0.4, np.inf]
            units = ''

        wr_data.dropna(inplace=True)

        # print(wr_data)

        total_count = wr_data.shape[0]
        calm_count = wr_data.query("WindVel == 0").shape[0]

        def speed_labels(bins, units):   
            labels = []
            for left, right in zip(bins[:-1], bins[1:]):
                if np.isinf(right):
                    labels.append('>{} {}'.format(left, units))
                elif np.isinf(left):
                    labels.append('<{} {}'.format(right, units))
                else:
                    labels.append('{} to {} {}'.format(left, right, units))

            return list(labels)

        def _convert_dir(directions, N=None):
            if N is None:
                N = directions.shape[0]
            barDir = directions * np.pi/180. - np.pi/N
            barWidth = 2 * np.pi / N
            return barDir, barWidth


        spd_labels = speed_labels(spd_bins, units=units)

        dir_bins = np.arange(-7.5, 370, 15)
        dir_labels = (dir_bins[:-1] + dir_bins[1:]) / 2

        rose = (
            wr_data.assign(WindVel_bins=lambda df:
                    pd.cut(df['WindVel'], bins=spd_bins, labels=spd_labels, right=True)
                 )
                .assign(WindDir_bins=lambda df:
                    pd.cut(df['WindDir'], bins=dir_bins, labels=dir_labels, right=False)
                 )
                .replace({'WindDir_bins': {360: 0}})
                .groupby(by=['WindVel_bins', 'WindDir_bins'])
                .size()
                .unstack(level='WindVel_bins')
                .fillna(0)
                .sort_index(axis=1)
                .applymap(lambda x: x / total_count * 100)
        )

#         print(rose)


        def wind_rose(rosedata, wind_dirs, palette=None, ax=ax):
            if palette is None:
                palette = seaborn.color_palette('inferno', n_colors=rosedata.shape[1])

            bar_dir, bar_width = _convert_dir(wind_dirs)
            
            if ax == None:
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                
            ax.set_theta_direction('clockwise')
            ax.set_theta_zero_location('N')
            ax.set_title(title)

            for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
                if n == 0:
                    # first column only
                    ax.bar(bar_dir, rosedata[c1].values, 
                           width=bar_width,
                           color=palette[0],
                           edgecolor='none',
                           label=c1,
                           linewidth=0)

                # all other columns
                ax.bar(bar_dir, rosedata[c2].values, 
                       width=bar_width, 
                       bottom=rosedata.cumsum(axis=1)[c1].values,
                       color=palette[n+1],
                       edgecolor='none',
                       label=c2,
                       linewidth=0)

            leg = ax.legend(loc=(0.17, -0.2), ncol=2)
                
            xt = ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
            xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

#             return fig


        directions = np.arange(0, 360, 15)
        fig = wind_rose(rose, directions)
#         plt.show()


    def weibull(self, height='H6', ax=None, return_weibull=False, bins=20):
        
        U = self.wind_speed[height]
        Ubar = np.nanmean(U)
        stdU = np.nanstd(U)
        
        k = (stdU/Ubar)**(-1.086) # taken from Sandrine's lecture on wind energy, p. 118
        c = Ubar/gamma(1+1/k)
        
        uu = np.linspace(0, 20, 50)
        ww = ((k/c)*(uu/c)**(k-1))*np.exp(-(uu/c)**k)
#         return uu, ww
        
        if ax == None:
            fig, ax = plt.subplots(1,1, figsize=[6,4])
        wb = ax.hist(U, bins=bins, label='_nolegend_', color='C0', alpha=0.6, density=True);
        ax.hist(U, bins=bins, label='_nolegend_', edgecolor='C0', histtype=u'step', linewidth=1, density=True)
        ax.plot(uu, ww, label=f'Weibull fit\nk = {k:.2f}\nc = {c:.2f} m/s', linewidth=2, c='C1')
        ax.set_xlabel('Wind speed [m/s]')
        ax.set_ylabel('Probability density')
        ax.legend()
        ax.set_title('Wind speed histogram')
        
        if return_weibull:
            return {'hist': wb, 'k': k, 'c': c}
#         plt.show()

    def weibull_and_windrose(self, fig=None, num_rows=1, row=0, return_weibull=False, bins=20):
        
        if fig == None:
            fig = plt.figure(figsize=[16,5])
            
        ax1 = fig.add_subplot(num_rows,2,2*row+1)
        ax2 = fig.add_subplot(num_rows,2,2*row+2, projection='polar')
#         plt.subplots_adjust(wspace=0.02, hspace=0.1)

        self.windrose(height='H6', ax=ax2)
        wb = self.weibull(ax=ax1, return_weibull=True, bins=bins)
        ax1.set_title('(a) Weibull distribution at hub height (91 m)')
        ax2.set_title('(b) Windrose at hub height (91 m)')
        
        if return_weibull:
            return wb


    def plot_velocity(self, ax=None):
        
        marg = min(self.heights)-(max(self.heights)-min(self.heights))*0.04
    
        kw = {"width_ratios":[95,3]}
        if ax == None:
            fig, ((axP, cax)) = plt.subplots(1,2, gridspec_kw=kw, figsize=[16,5])
        else:
            axP, cax = ax
            
        plt.subplots_adjust(wspace=0.02)
        
        X, Y = np.meshgrid(self.wind_speed.index, self.heights)
        axP.grid(visible=1)
        Z = self.wind_speed
        im = axP.pcolormesh(X, Y, np.transpose(Z), cmap='plasma', shading='auto', zorder=1)
        axP.set_ylabel('Height [m]')
        axP.set_title('Wind speed')
        axP.set_ylim(bottom=marg)
        axP.fill_between(self.wind_speed.index, marg, min(self.heights), where=~self.daylight, 
                         color='midnightblue', alpha=0.7, zorder=0)
        cbar = plt.colorbar(im, cax=cax, label='Wind speed [m/s]')
        
    def plot_velocity_dir(self, ax=None):
        
        marg = min(self.heights)-(max(self.heights)-min(self.heights))*0.04
    
        kw = {"width_ratios":[95,3]}
        if ax == None:
            fig, ((axP, cax)) = plt.subplots(1,2, gridspec_kw=kw, figsize=[16,5])
        else:
            axP, cax = ax
            
        plt.subplots_adjust(wspace=0.02)
        
        X, Y = np.meshgrid(self.wind_speed.index, self.heights)
        D1 = np.sin(self.wind_direction/180*np.pi)
        D2 = np.cos(self.wind_direction/180*np.pi)
        axP.quiver([X,Y],np.transpose(D1),np.transpose(D2))
        #axP.grid(visible=1)
        #Z = self.wind_speed
        #im = axP.pcolormesh(X, Y, np.transpose(Z), cmap='plasma', shading='auto', zorder=1)
        #axP.set_ylabel('Height [m]')
        #axP.set_title('Wind speed')
        #axP.set_ylim(bottom=marg)
        #axP.fill_between(self.wind_speed.index, marg, min(self.heights), where=~self.daylight, 
        #                 color='midnightblue', alpha=0.7, zorder=0)
        #cbar = plt.colorbar(im, cax=cax, label='Wind speed [m/s]')
        
    def plot_direction(self, ax=None):
        
        marg = min(self.heights)-(max(self.heights)-min(self.heights))*0.04
    
        kw = {"width_ratios":[95,3]}
        if ax == None:
            fig, ((axP, cax)) = plt.subplots(1,2, gridspec_kw=kw, figsize=[16,5])
        else:
            axP, cax = ax
            
        plt.subplots_adjust(wspace=0.02)
        
        X, Y = np.meshgrid(self.wind_direction.index, self.heights)
        axP.grid(visible=1)
        Z = self.wind_direction
        im = axP.pcolormesh(X, Y, np.transpose(Z), cmap='hsv', shading='auto', zorder=1)
        axP.set_ylabel('Height [m]')
        axP.set_title('Wind Direction')
        axP.set_ylim(bottom=marg)
        axP.fill_between(self.wind_direction.index, marg, min(self.heights), where=~self.daylight, 
                         color='midnightblue', alpha=0.7, zorder=0)
        cbar = plt.colorbar(im, cax=cax, label='Wind direction')
        
    def plot_direction_2(self, ax=None):
        
        plt.plot(self.wind_direction.index, self.wind_direction)
        plt.legend()

    
        
    def plot_television(self, ax=None):
        
        get_time_of_day = lambda date: date.hour + date.minute/60
        get_current_day = lambda date: date.day
        
        eradata = self.ERA[['current_day', 'time_of_day', 'cloud_cover']]
        cloud_cover = eradata.groupby(by=['current_day', 'time_of_day']).mean().unstack(level='time_of_day')
        cloud_cover.index = eradata.current_day.unique()
        cloud_cover.columns = eradata.time_of_day.unique()
        
        time_of_day = self.wind_speed.index.to_series().apply(get_time_of_day)
        current_day = self.wind_speed.index.to_series().apply(get_current_day)
        
        time_of_day_llj = self.llj.index.to_series().apply(get_time_of_day)
        current_day_llj = self.llj.index.to_series().apply(get_current_day)
        
        no_data = (pd.DataFrame({'reliable': self.reliable_profile, 'current_day': current_day, 'time_of_day': time_of_day})
           .groupby(by=['current_day', 'time_of_day'])
           .mean()
           .unstack(level='time_of_day')
           .replace({1: np.nan, 0: 1})
           .set_index(current_day.unique()))
        
        no_data.columns = [round(i,2) for i in time_of_day.unique()]

        kw = {"width_ratios":[92, 3, 3]}
        fig0, (ax, cax1, cax2) = plt.subplots(1,3, gridspec_kw=kw, figsize=[15,7])
        cloud = ax.pcolormesh(cloud_cover.index, cloud_cover.columns+0.5, np.transpose(cloud_cover),
                              cmap='Greys', alpha=0.5, shading='auto', zorder=1)
        nodata = ax.pcolormesh(no_data.index, no_data.columns, np.transpose(no_data),
                              cmap='Greys', alpha=1, shading='auto', zorder=2,
                              vmin=0, vmax=1.2, hatch='//')
        lljs = ax.scatter(current_day_llj, time_of_day_llj, c=self.llj.speed, cmap='plasma', zorder=3, s=50)
        ax.set_ylim([0,24])
        ax.set_ylabel('Hour of the day')
        ax.set_xlabel('Day of the month')
        plt.colorbar(cloud, cax=cax1, label='Cloud cover')
        plt.colorbar(lljs, cax=cax2, label='LLJ speed [m/s]')

        plt.show()
        
        
    def plot_llj_distribution(self):
        
        def v2c(value, cmap_name='YlGnBu', vmin=0, vmax=1):
            # honestly stolen from stack overflow
            # norm = plt.Normalize(vmin, vmax)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap(cmap_name)  # PiYG
            rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
            color = mpl.colors.rgb2hex(rgb)
            return color
    
        nose_h = sorted(self.llj.height.unique())
        colors = [v2c(val, cmap_name='plasma', vmin=min(nose_h), vmax=max(nose_h)) for val in nose_h]
        
        get_time_of_day = lambda date: date.hour + date.minute/60
        get_current_day = lambda date: date.day
        
        llj = self.llj
        llj['time_of_day'] = llj.index.to_series().apply(get_time_of_day)
        
        kw = {"width_ratios":[5,6]}
        fig, ax = plt.subplots(1,2, figsize=[16,5], gridspec_kw=kw)
        
#         h = ax.hist([self.llj[self.llj.height==hh].time_of_day for hh in nose_h],
#                      bins=24, histtype='barstacked', label=['h = '+str(hh)+' m' for hh in nose_h],
#                      color=colors)
#         ax[0].legend(ncol=2)
        ax[0].hist(llj.time_of_day, bins=24)
        ax[0].set_xlabel('Hours of the day')
        ax[0].set_ylabel('Number of cases')
        ax[0].set_title(f'LLJ occurrence vs time of the day')
#         plt.colorbar(colors)


        im = ax[1].scatter(llj.speed, llj.direction, c=llj.height, s=15, alpha=0.6, cmap='plasma')
        ax[1].set_xlabel('LLJ core speed [m/s]')
        ax[1].set_ylabel('LLJ core direction [deg]')
        ax[1].grid(visible=1)
        plt.colorbar(im, ax=ax[1], label='LLJ core height [m]')
        
    def llj_square(self, ax=None, cbar=False, cax=None,
                   xlabel='Day of the month', ylabel='Hour of the day', title='LLJ occurence', vmin=None, vmax=None):
        
        if ax == None:
            fig, ax = plt.subplots(1,1, figsize=[6,5])
        
        get_time_of_day = lambda date: date.hour + date.minute/60
        get_current_day = lambda date: date.day
        
        llj = self.llj
        llj['time_of_day'] = llj.index.to_series().apply(get_time_of_day)
        llj['current_day'] = llj.index.to_series().apply(get_current_day)
        
        im = ax.scatter(llj.current_day, llj.time_of_day, c=llj.speed, alpha=0.4, cmap='plasma', vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim([0,32])
        ax.grid(visible=1)
        
        if cbar:
            if cax == None: plt.colorbar(im, ax=ax)
            else: plt.colorbar(im, cax=cax, label='LLJ core speed [m/s]')
        
        
    def analyse_stability(self, show_plot=False, show_map=False, show_pie=False, classify=False, pie_loc='C',
                       stability_file=r'C:\Users\barma\Desktop\REM\Master thesis\code\data\nc_scans\stability.nc'):
        
        cases = ['L', 'C', 'S1', 'S2', 'S3']
        st = self.wind_speed.index.min()
        en = self.wind_speed.index.max()
        
        try:
            ss = nc.Dataset(stability_file, 'r', format='NETCDF4')
            sstime = nc.num2date(ss.variables['time'][:], ss.variables['time'].units,
                                 only_use_cftime_datetimes=False,
                                 only_use_python_datetimes=True).data
            zeta = pd.DataFrame(ss.variables['zeta'][:,:].data, index=sstime, columns=cases)
            self.zeta = zeta.loc[st:en]
            zeta_classes = pd.DataFrame(ss.variables['zeta_class'][:,:].data, index=sstime, columns=cases)
            self.zeta_classes = zeta_classes.loc[st:en]
            coords = pd.DataFrame({'lat': ss.variables['latitude'][:], 'lon': ss.variables['longitude'][:]}, index=cases)
            daylight = pd.Series(ss.variables['daylight'][:].data.astype('bool'), index=sstime).loc[st:en]
        finally:
            ss.close()
        
        if show_pie:
            
            
            unstable = (self.zeta[pie_loc] < -0.2).sum()*100/len(self.zeta[pie_loc])
            stable = (self.zeta[pie_loc] > 0.2).sum()*100/len(self.zeta[pie_loc])
            neutral = ((self.zeta[pie_loc] >= -0.2) & (self.zeta[pie_loc] <= 0.2)).sum()*100/len(self.zeta[pie_loc])

            fig, ax = plt.subplots(1,1,figsize=[7,5])
            # ax.scatter(scan.shear.local_exp, scan.zeta.C)
            ax.pie([stable, neutral, unstable], labels=['Stable', 'Neutral', 'Unstable'], autopct='%1.1f%%',
                    colors=['cornflowerblue', 'papayawhip', 'lightcoral'], shadow=True, startangle=90)
        
        if show_map:
            
            lons = [-3.0, -2.75, -2.5, -2.25, -2.0]
            lats = [48.0, 47.75, 47.5, 47.25, 47.0]
            
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1,1,1)
            from mpl_toolkits.basemap import Basemap
            VMM_lon, VMM_lat = (-2.516, 47.2755)

            map1 = Basemap(projection='merc', resolution='h',
                             llcrnrlat=46.9,
                             llcrnrlon=-2.9,
                             urcrnrlat=47.6,
                             urcrnrlon=-1.9)

            map1.drawcoastlines(linewidth=1)
            map1.fillcontinents(color='coral',lake_color='aqua')
            map1.drawmapboundary(fill_color='aqua')
            map1.drawmeridians(np.arange(-4, -1, 0.2), labels=[True, False, True])
            map1.drawparallels(np.arange(47,48,0.2), labels=[True, False])

            for lon in lons:
                for lat in lats:
                    x, y = map1(lon, lat)
                    map1.plot(x,y, marker='+', color='black')

            x0, y0 = map1(VMM_lon, VMM_lat)
            map1.plot(x0, y0, marker='D', color='red')
            plt.annotate('VMM', xy=(x0, y0),  xycoords='data',
                            xytext=(-10, 20), textcoords='offset points',
                            color='r',
                            arrowprops=dict(arrowstyle='->')
                            )

            for cc in cases:
                xl, yl = map1(coords.loc[cc,'lon'], coords.loc[cc, 'lat']-0.04)
                plt.text(xl, yl, cc, fontsize=13, fontweight='bold')
                
        if classify:
            zdata = self.zeta_classes
            clabel = 'Stability class'
        else:
            zdata = self.zeta
            clabel = 'Stability parameter'     
        
        if show_plot:
            kw = {"height_ratios": [5,2], "width_ratios": [98,2]}
            fig, ax = plt.subplots(2,2, figsize=[18,7], gridspec_kw=kw)
            ax[0][0].set_title('Atmospheric stability in the region')
            plt.subplots_adjust(wspace=0.02, hspace=0.05)
            ax[0][0].get_shared_x_axes().join(ax[0][0], ax[1][0])
            ax[0][0].set_xlim(left=zdata.index[0], right=zdata.index[-1])

            z = ax[0][0].pcolormesh(zdata.index, zdata.columns, zdata.transpose(), shading='auto', cmap='RdBu')
            ax[0][0].set_xticks([])
            ax[0][0].hlines([0.5, 1.5, 2.5, 3.5], colors='black', linestyles='solid',
                            linewidth=4.5, xmin=min(zdata.index), xmax=max(zdata.index))
            ax[0][0].hlines([0.5, 1.5, 2.5, 3.5], colors='white', linestyles='solid',
                            linewidth=3, xmin=min(zdata.index), xmax=max(zdata.index))
            plt.colorbar(z, label=clabel, cax=ax[0][1])

            l = ax[1][0].scatter(self.llj.index, self.llj.speed, c=self.wind_direction.H2.loc[self.llj.index],
                                 cmap='twilight_shifted', zorder=1)
            ax[1][0].fill_between(zdata.index, 0, 1, where=~daylight,
                            color='indigo', alpha=0.1, transform=ax[1][0].get_xaxis_transform(), zorder=0)
            plt.colorbar(l, label='H2 wind direction [deg]', cax=ax[1][1])
            ax[1][0].set_ylabel('LLJ core speed [m/s]')

        
    def show_profile(self, timestamp):
        
        profile = self.wind_speed.loc[timestamp]
        
        fig, ax = plt.subplots(1,1, figsize=[5,8])
        ax.plot(profile, self.heights, '-o')
        ax.set_xlabel('Wind speed [m/s]')
        ax.set_ylabel('Height AGL [m]')
        ax.set_xlim([0,25])
        ax.grid(visible=1)
        
        
    def show_shear_profile(self, timestamp):
        
        Us = self.wind_speed.loc[timestamp]
        (slope, intercept, r, p, se) = linregress(np.log(np.array(self.heights, dtype='float32')), np.log(Us))
        
        kw = {'width_ratios': [4, 9]}
        fig, ax = plt.subplots(1,2, figsize=[14,8], gridspec_kw=kw)
        ax[0].plot(Us, self.heights, '-o')
        ax[0].set_xlabel('Wind speed [m/s]')
        ax[0].set_ylabel('Height AGL [m]')
        ax[0].set_xlim([0,25])
        ax[0].grid(visible=1)
        ax[0].set_title(timestamp)
        
        ax[1].scatter(np.log(np.array(self.heights[0:10], dtype='float32')/self.heights[0]), np.log(Us[0:10]/Us[0]))
        ax[1].set_xlabel('Log height')
        ax[1].set_ylabel('Log wind speed')
        ax[1].grid(visible=1)
        
        print(f'Wind shear exponent: {self.shear.loc[timestamp].exp:.3f}')
        print(f'Wind shear r2: {self.shear.loc[timestamp].r2:.3f}')
        print(f'Wind shear across rotor: {self.shear.loc[timestamp].rotor:.3f}')
        print(f'Wind shear exponent at hub height: {self.shear.loc[timestamp].local_exp:.3f}')
        
#         uu = (Us[5]+Us[6])/2
#         zz = (self.heights[5]+self.heights[6])/2
#         new_sh = ((Us[6]-Us[5])/(self.heights[6]-self.heights[5]))*(zz/uu)
#         print(f'New shear: {new_sh:.3f}')
        

        
    
    
    
    
    
    
    
    
    

