# ECOSTRESS Level 3 and 4 Evapotranspiration, Evaporative Stress Index, Water Use Efficiency

[![CI](https://github.com/ECOSTRESS-Collection-3/ECOv003-L3T-L4T-JET/actions/workflows/ci.yml/badge.svg)](https://github.com/ECOSTRESS-Collection-3/ECOv003-L3T-L4T-JET/actions/workflows/ci.yml)

This is the main repository for the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) collection 3 level 3/4 evapotranspiration data products algorithm.

The ECOSTRESS collection 3 level 3/4 evapotranspiration data products algorithm is the pre-cursor to the [Surface Biology and Geology (SBG) collection 1 level 4 evapotranspiration data products algorithm](https://github.com/sbg-tir/SBG-TIR-L4-JET).

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Kerry Cawse-Nicholson](https://github.com/kcawse) (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Madeleine Pascolini-Campbell](https://github.com/madeleinepc) (she/her)<br>
[madeleine.a.pascolini-campbell@jpl.nasa.gov](mailto:madeleine.a.pascolini-campbell@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329F

[Evan Davis](https://github.com/evandjpl) (he/him)<br>
[evan.w.davis@jpl.nasa.gov](mailto:evan.w.davis@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 397K

[Claire Villanueva-Weeks](https://github.com/clairesvw) (she/her)<br>
NASA Jet Propulsion Laboratory 329G

The code for the ECOSTRESS level 3 JET PGE has been developed using open-science practices based on the [ECOSTRESS Collection 2 JET PGE](https://github.com/ECOSTRESS-Collection-2/ECOv002-L3T-L4T-JET) with New Technology Report (NTR) and open-source license from NASA Jet Propulsion Laboratory.

The repositories for the evapotranspiration algorithms are located in the [JPL-Evapotranspiration-Algorithms](https://github.com/JPL-Evapotranspiration-Algorithms) organization.

## 1. Introduction 

This software produces estimates of:
- evapotranspiration (ET)
- evaporative stress index (ESI)
- water use efficiency (WUE)

Evapotranspiration (ET) is one of the main science outputs from the ECOSTRESS mission. ET is a Level-3 (L3) product constructed from a combination of the ECOSTRESS Level-2 (L2) Land Surface Temperature & Emissivity (LSTE) product and auxiliary data sources. Accurate modelling of ET requires consideration of many environmental and biological controls including: solar radiation, the atmospheric water vapor deficit, soil water availability, vegetation physiology and phenology (Brutsaert, 1982; Monteith, 1965; Penman, 1948). Scientists develop models that ingest global satellite observations to capture these environmental and biological controls on ET. LST holds the unique ability to capture when and where plants experience stress, as observed by elevated temperatures which can idenitfy areas that have a reduced capacity to evaporate or transpire water to the atmosphere (Allen et al., 2007). The ECOSTRESS evapotranspiration product combines the [surface temperature and emissivity observations from the ECOSTRESS sensor](https://github.com/ECOSTRESS-Collection-3/ECOv003-L2-LSTE) with the [NDVI and albedo estimated by STARS](https://github.com/ECOSTRESS-Collection-3/ECOv003-L2-STARS), estimates near-surface meteorology by downscaling GEOS-5 FP to these three high resolution images, and runs these variables through a set of surface energy balance models.

The repositories for the evapotranspiration algorithms are located in the [JPL-Evapotranspiration-Algorithms](https://github.com/JPL-Evapotranspiration-Algorithms) organization.

## 2. Data Products 

### 2.1. Metadata

Metadata for the ECOSTRESS Collection 3 tiled products is provided in JSON format:
```
{
  "StandardMetadata": {
    "AuxiliaryInputPointer": "AuxiliaryNWP",
    "AutomaticQualityFlag": "PASS",
    "AutomaticQualityFlagExplanation": "Image matching performed to correct orbit ephemeris/attitude",
    "BuildID": "0800",
    "CRS": "+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +type=crs",
    "CampaignShortName": "Primary",
    "CollectionLabel": "ECOv003",
    "DataFormatType": "COG",
    "DayNightFlag": "Day",
    "EastBoundingCoordinate": -114.74406148409601,
    "ImageLineSpacing": 70.0,
    "ImageLines": 1568,
    "ImagePixelSpacing": 70.0,
    "ImagePixels": 1568,
    "InputPointer": "ECOv003_L2_LSTE_21485_013_20220420T211350_0800_02.h5,ECOv003_L2_CLOUD_21485_013_20220420T211350_0800_02.h5,ECOSTRESS_L1B_GEO_21485_013_20220420T211350_0800_01.h5,ECOv003_L1B_RAD_21485_013_20220420T211350_0800_01.h5",
    "InstrumentShortName": "ECOSTRESS",
    "LocalGranuleID": "ECOv003_L3T_JET_21485_013_11SPS_20220420T211350_0800_01.zip",
    "LongName": "ECOSTRESS Tiled Evapotranspiration Ensemble Instantaneous and Daytime L3 Global 70 m",
    "NorthBoundingCoordinate": 33.43490918842431,
    "PGEName": "L3T_L4T_JET",
    "PGEVersion": "v1.10.12",
    "PlatformLongName": "ISS",
    "PlatformShortName": "ISS",
    "PlatformType": "Spacecraft",
    "ProcessingEnvironment": "Linux eco-p44.tir 3.10.0-1160.45.1.el7.x86_64 #1 SMP Wed Oct 13 17:20:51 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux",
    "ProcessingLevelDescription": "Level 3 Tiled Evapotranspiration Ensemble",
    "ProcessingLevelID": "L3T",
    "ProducerAgency": "JPL",
    "ProducerInstitution": "Caltech",
    "ProductionDateTime": "2022-04-22T10:47:36.452Z",
    "ProductionLocation": "ECOSTRESS Science Computing Facility",
    "RangeBeginningDate": "2022-04-20",
    "RangeBeginningTime": "21:13:51.290937",
    "RangeEndingDate": "2022-04-20",
    "RangeEndingTime": "21:18:51.290937",
    "RegionID": "11SPS",
    "SISName": "Level 3/4 JET Product Specification Document",
    "SISVersion": "Preliminary",
    "SceneBoundaryLatLonWKT": "POLYGON ((-118.30553175600564 30.910805591562212, -115.4606649798732 33.891051544444885, -112.80123658401774 31.518067522267156, -115.65618262396845 28.61321628442961, -118.30553175600564 30.910805591562212))",
    "SceneID": "13",
    "ShortName": "ECO_L3T_JET",
    "SouthBoundingCoordinate": 32.42972726825087,
    "StartOrbitNumber": "21485",
    "StopOrbitNumber": "21485",
    "WestBoundingCoordinate": -115.9361545299493
  },
  "ProductMetadata": {
    "BandSpecification": [
      0.0,
      0.0,
      8.779999732971191,
      0.0,
      10.520000457763672,
      12.0
    ],
    "NumberOfBands": 3,
    "OrbitCorrectionPerformed": "True",
    "QAPercentCloudCover": 0.009151460329029571,
    "QAPercentGoodQuality": 98.34781568877551,
    "AuxiliaryNWP": "GEOS.fp.asm.inst3_2d_asm_Nx.20220420_2100.V01.nc4,GEOS.fp.asm.inst3_2d_asm_Nx.20220421_0000.V01.nc4,GEOS.fp.asm.tavg1_2d_lnd_Nx.20220420_2030.V01.nc4,GEOS.fp.asm.tavg1_2d_lnd_Nx.20220420_2130.V01.nc4,GEOS.fp.asm.tavg1_2d_rad_Nx.20220420_2030.V01.nc4,GEOS.fp.asm.tavg1_2d_rad_Nx.20220420_2130.V01.nc4,GEOS.fp.asm.tavg1_2d_slv_Nx.20220420_2030.V01.nc4,GEOS.fp.asm.tavg1_2d_slv_Nx.20220420_2130.V01.nc4,GEOS.fp.asm.tavg3_2d_aer_Nx.20220420_1930.V01.nc4,GEOS.fp.asm.tavg3_2d_aer_Nx.20220420_2230.V01.nc4,GEOS.fp.asm.tavg3_2d_chm_Nx.20220420_1930.V01.nc4,GEOS.fp.asm.tavg3_2d_chm_Nx.20220420_2230.V01.nc4"
  }
}
```

#### 2.1.1. Standard Metadata
Information on the `StandardMetadata` is included on the [ECOSTRESS GitHub landing page](https://github.com/ECOSTRESS-Collection-3)

#### 2.1.2. Product Metadata

| **Name** | **Type** |
| --- | --- |
| BandSpecification | float |
| NumberOfBands | integer |
| OrbitCorrectionPerformed | string |
| QAPercentCloudCover | float |
| QAPercentGoodQuality | float |
| AuxiliaryNWP | string |

*Table 9. Name and type of metadata fields contained in the common ProductMetadata group in each L2T/L3T/L4T product.*

| **Product Long Name** | **Product Short Name** |
| --- | --- |
| STARS NDVI/Albedo | L2T STARS |
| Ecosystem Auxiliary Inputs | L3T ETAUX |
| Evapotranspiration | L3T JET |
| Evaporative Stress Index | L4T ESI |
| Water Use Efficiency | L4T WUE |

*Table 1. Listing of ECOSTRESS ecosystem products long names and short names.*

### 2.2. Quality Flags

Two high-level quality flags are provided in all gridded and tiled products as thematic/binary masks encoded to zero and one in unsigned 8-bit integer layers. The cloud layer represents the final cloud test from L2 CLOUD. The water layer represents the surface water body in the Shuttle Radar Topography Mission (SRTM) Digital Elevation Model. For both layers, zero means absence, and one means presence. Pixels with the value 1 in the cloud layer represent detection of cloud in that pixel. Pixels with the value 1 in the water layer represent open water surface in that pixel. All tiled product data layers written in `float32` contain a standard not-a-number (`NaN`) value at each pixel that could not be retrieved. The cloud and water layers are provided to explain these missing values.

### 2.3. L3T ETAUX Ecosystem Auxiliary Inputs Product

The ECOSTRESS ecosystem processing chain is designed to be independently reproducible. To facilitate open science, the auxiliary data inputs that are produced for evapotranspiration processing are distributed as a data product, such that the end user has the ability to run their own evapotranspiration model using ECOSTRESS data. The data layers of the L3T ETAUX product are described in Table 2.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| Ta | Near-surface air temperature | float32 | Celsius | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| RH | Relative Humidity | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| SM | Soil Moisture | float32 | m³/m³ | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| Rg | Global Radiation | float32 | W/m^2 | NaN | N/A | 0 | N/A | N/A | 12.06 mb |
| Rn | Net Radiation | float32 | Ratio | NaN | N/A | 0 | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

*Table 2. Listing of the L3T ETAUX data layers.*

### 2.4. Downscaled Meteorology & Soil Moisture

```mermaid
flowchart TB
    subgraph ECO_L2[ECOSTRESS L2]
        direction TB
        ECO_L2T_STARS[ECOSTRESS<br>L2T_STARS<br>NDVI<br>&<br>Albedo<br>Product]
        ECO_L2T_LSTE[ECOSTRESS<br>L2T_LSTE<br>Surface Temperature<br>&<br>Emissivity<br>Product]
        ST[Surface Temperature 70m]
        NDVI[NDVI 70m]
        albedo[Albedo 70m]
        ECO_L2T_LSTE --> ST
        ECO_L2T_STARS --> NDVI
        ECO_L2T_STARS --> albedo
    end

    subgraph GEOS5FP[GEOS-5 FP]
        direction TB
        GEOS5FP_Ta[GEOS-5 FP<br>Air<br>Temperature]
        GEOS5FP_RH[GEOS-5 FP<br>Humidity]
        GEOS5FP_SM[GEOS-5 FP<br>Soil<br>Moisture]
    end

    subgraph downscaling[Downscaling]
        direction TB
        downscale_Ta[Air<br>Temperature<br>Downscaling]
        downscale_RH[Humidity<br>Downscaling]
        downscale_SM[Soil<br>Moisture<br>Downscaling]
    end

    subgraph downscaled_meteorology[Downscaled Meteorology]
        direction TB
        downscaled_Ta[Downscaled<br>70m<br>Air<br>Temperature]
        downscaled_RH[Downscaled<br>70m<br>Humidity]
        downscaled_SM[Downscaled<br>70m<br>Soil<br>Moisture]
    end

    GEOS5FP_Ta --> downscale_Ta
    ST --> downscale_Ta
    NDVI --> downscale_Ta
    albedo --> downscale_Ta

    GEOS5FP_RH --> downscale_RH
    ST --> downscale_RH
    NDVI --> downscale_RH
    albedo --> downscale_RH

    GEOS5FP_SM --> downscale_SM
    ST --> downscale_SM
    NDVI --> downscale_SM
    albedo --> downscale_SM

    downscale_Ta --> downscaled_Ta
    downscale_RH --> downscaled_RH
    downscale_SM --> downscaled_SM
```

Coarse resolution near-surface air temperature (Ta) and relative humidity (RH) are taken from the GEOS-5 FP `tavg1_2d_slv_Nx` product. Ta and RH are down-scaled using a linear regression between up-sampled ST, NDVI, and albedo as predictor variables to Ta or RH from GEOS-5 FP as a response variable, within each Sentinel tile. These regression coefficients are then applied to the 70 m ST, NDVI, and albedo, and this first-pass estimate is then bias-corrected to the coarse image from GEOS-5 FP. These downscaled meteorology estimates are recorded in the L3T ETAUX product listed in Table 2. Areas of cloud are filled in with bi-cubically resampled GEOS-5 FP. This same down-scaling procedure is applied to soil moisture (SM) from the GEOS-5 FP `tavg1_2d_lnd_Nx` product, which is recorded in the L3T ETAUX product listed in Table 2.

### 2.5. Surface Energy Balance

```mermaid
flowchart TB
    subgraph ECO_L2[ECOSTRESS L2]
        direction TB
        ECO_L2T_STARS[ECOSTRESS<br>L2T_STARS<br>NDVI<br>&<br>Albedo<br>Product]
        ECO_L2T_LSTE[ECOSTRESS<br>L2T_LSTE<br>Surface Temperature<br>&<br>Emissivity<br>Product]
        ST[Surface Temperature 70m]
        NDVI[NDVI 70m]
        albedo[Albedo 70m]
        ECO_L2T_LSTE --> ST
        ECO_L2T_STARS --> NDVI
        ECO_L2T_STARS --> albedo
    end

    subgraph downscaled_meteorology[Downscaled Meteorology]
        direction TB
        downscaled_Ta[Downscaled<br>70m<br>Air<br>Temperature]
        downscaled_RH[Downscaled<br>70m<br>Humidity]
        downscaled_SM[Downscaled<br>70m<br>Soil<br>Moisture]
    end

    subgraph GEOS5FP[GEOS-5 FP]
        direction TB
        GEOS5FP_AOT[GEOS-5 FP AOT]
        GEOS5FP_COT[GEOS-5 FP COT]
    end

    BESS_JPL_Rn[BESS-JPL<br>70m<br>Net<br>Radiation]
    BESS_JPL_GPP[BESS-JPL<br>70m<br>GPP]
    BESS_JPL_ET[BESS-JPL<br>70m<br>ET]

    GEOS5FP_AOT --> FLiES
    GEOS5FP_COT --> FLiES
    albedo --> FLiES
    
    FLiES --> BESS_JPL
    ST --> BESS_JPL
    NDVI --> BESS_JPL
    albedo --> BESS_JPL
    downscaled_Ta --> BESS_JPL
    downscaled_RH --> BESS_JPL
    downscaled_SM --> BESS_JPL

    BESS_JPL --> BESS_JPL_Rn
    BESS_JPL --> BESS_JPL_GPP
    BESS_JPL --> BESS_JPL_ET
```

The surface energy balance processing for ECOSTRESS begins with an artificial neural network (ANN) implementation of the Forest Light Environmental Simulator (FLiES) radiative transfer algorithm, following the workflow established by Dr. Hideki Kobayashi and Dr. Youngryel Ryu. GEOS-5 FP provides sub-daily Cloud Optical Thickness (COT) in the `tavg1_2d_rad_Nx` product and Aerosol Optical Thickness (AOT) from `tavg3_2d_aer_Nx`. Together with STARS albedo, these variables are run through the ANN implementation of FLiES to estimate incoming shortwave radiation (Rg), bias-corrected to Rg from the GEOS-5 FP `tavg1_2d_rad_Nx` product.

The Breathing Earth System Simulator-Jet Propulsion Laboratory (BESS-JPL) algorithm, contributed by Dr. Youngryel Ryu, iteratively calculates net radiation (Rn), ET, and Gross Primary Production (GPP) estimates. The BESS-JPL Rn is used as the Rn input to the remaining ET models and is recorded in the L3T ETAUX product listed in Table 2.

### 2.6. L3T JET Evapotranspiration Product

The ECOSTRESS Collection 3 L3T JET product uses an ensemble of four evapotranspiration models (PT-JPL-SM, STIC-JPL, PM-JPL, and BESS-JPL) to produce robust evapotranspiration estimates. The data layers include individual model outputs, ensemble statistics, canopy partitioning components, and quality flags. The data layers for the L3T JET products are listed in Table 3.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| PTJPLSMinst | PT-JPL-SM Instantaneous | float32 | W/m^2 | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| PTJPLSMdaily | PT-JPL-SM Daily | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| STICJPLdaily | STIC-JPL Daily | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| BESSJPLdaily | BESS-JPL Daily | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| PMJPLdaily | PM-JPL (MOD16) Daily | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| ETdaily | Daily Evapotranspiration | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| ETinstUncertainty | Instantaneous Evapotranspiration Uncertainty | float32 | W/m^2 | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| PTJPLSMcanopy | PT-JPL-SM Canopy | float32 | proportion | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| STICJPLcanopy | STIC-JPL Canopy | float32 | proportion | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| PTJPLSMsoil | PT-JPL-SM Soil | float32 | proportion | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| PTJPLSMinterception | PT-JPL-SM Interception | float32 | proportion | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

*Table 3. Listing of the L3T JET data layers.*

Following design of the L3T JET product from ECOSTRESS Collection 2, the ECOSTRESS Collection 3 L3T JET product uses an ensemble of evapotranspiration models to produce robust evapotranspiration estimates. The ensemble approach combines outputs from four distinct models, each with different strengths and theoretical foundations, to reduce uncertainty and improve overall accuracy.

#### 2.6.1. Priestley-Taylor (PT-JPL-SM) Evapotranspiration Model

The Priestley-Taylor Jet Propulsion Laboratory model with Soil Moisture (PT-JPL-SM), developed by Dr. Adam Purdy and Dr. Joshua Fisher, was designed as a soil moisture-sensitive evapotranspiration product for the Soil Moisture Active-Passive (SMAP) mission. The model estimates instantaneous canopy transpiration, leaf surface evaporation, and soil moisture evaporation using the Priestley-Taylor formula with a set of constraints. These three partitions are combined into total latent heat flux in watts per square meter for the ensemble estimate.

**Reference**: Purdy, A.J., Fisher, J.B., Goulden, M.L., Colliander, A., Halverson, G., Tu, K., Famiglietti, J.S. (2018). SMAP soil moisture improves global evapotranspiration. *Remote Sensing of Environment*, 219, 1-14. https://doi.org/10.1016/j.rse.2018.09.023

**Repository**: [PT-JPL-SM](https://github.com/JPL-Evapotranspiration-Algorithms/PT-JPL-SM)

#### 2.6.2. Surface Temperature Initiated Closure (STIC-JPL) Evapotranspiration Model

The Surface Temperature Initiated Closure-Jet Propulsion Laboratory (STIC-JPL) model, contributed by Dr. Kaniska Mallick, was designed as a surface temperature-sensitive ET model, adopted by ECOSTRESS and SBG for improved estimates of ET reflecting mid-day heat stress. The STIC-JPL model estimates total latent heat flux directly using thermal remote sensing observations. This instantaneous estimate of latent heat flux is included in the ensemble estimate.

**Reference**: Mallick, K., Trebs, I., Boegh, E., Giustarini, L., Schlerf, M., Drewry, D.T., Hoffmann, L., von Randow, C., Kruijt, B., Araùjo, A., Saleska, S., Ehleringer, J.R., Domingues, T.F., Ometto, J.P.H.B., Nobre, A.D., de Moraes, O.L.L., Hayek, M., Munger, J.W., Wofsy, S.C. (2016). Canopy-scale biophysical controls of transpiration and evaporation in the Amazon Basin. *Hydrology and Earth System Sciences*, 20, 4237-4264. https://doi.org/10.5194/hess-20-4237-2016

**Repository**: [STIC-JPL](https://github.com/JPL-Evapotranspiration-Algorithms/STIC-JPL)

#### 2.6.3. Penman Monteith (PM-JPL) Evapotranspiration Model

The Penman-Monteith-Jet Propulsion Laboratory (PM-JPL) algorithm is a derivation of the MOD16 algorithm that was originally designed as the ET product for the Moderate Resolution Imaging Spectroradiometer (MODIS) and continued as a Visible Infrared Imaging Radiometer Suite (VIIRS) product. PM-JPL uses a similar approach to PT-JPL and PT-JPL-SM to independently estimate vegetation and soil components of instantaneous ET, but using the Penman-Monteith formula instead of the Priestley-Taylor. The PM-JPL latent heat flux partitions are summed to total latent heat flux for the ensemble estimate.

**Reference**: Running, S., Mu, Q., Zhao, M., Moreno, A. (2019). MODIS Global Terrestrial Evapotranspiration (ET) Product (MOD16A2/A3 and MOD16A2GF/A3GF). NASA Earth Observing System Data and Information System (EOSDIS) Land Processes Distributed Active Archive Center (LP DAAC). https://doi.org/10.5067/MODIS/MOD16A2.061

**Repository**: [PM-JPL](https://github.com/JPL-Evapotranspiration-Algorithms/PM-JPL)

#### 2.6.4. Breathing Earth System Simulator (BESS-JPL) Gross Primary Production (GPP) Model

The Breathing Earth System Simulator Jet Propulsion Laboratory (BESS-JPL) model is a coupled surface energy balance and photosynthesis model contributed by Dr. Youngryel Ryu. The model iteratively calculates net radiation, ET, and Gross Primary Production (GPP) estimates. The latent heat flux component of BESS-JPL is included in the ensemble estimate, while the BESS-JPL net radiation is used as input to the other ET models.

**Reference**: Ryu, Y., Baldocchi, D.D., Kobayashi, H., van Ingen, C., Li, J., Black, T.A., Beringer, J., van Gorsel, E., Knohl, A., Law, B.E., Roupsard, O. (2011). Integration of MODIS land and atmosphere products with a coupled-process model to estimate gross primary productivity and evapotranspiration from 1 km to global scales. *Global Biogeochemical Cycles*, 25, GB4017. https://doi.org/10.1029/2011GB004053

**Repository**: [BESS-JPL](https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL)

#### 2.6.5. Ensemble Processing

The median of total latent heat flux in watts per square meter from the PT-JPL-SM, STIC-JPL, PM-JPL, and BESS-JPL models is upscaled to a daily ET estimate in millimeters per day and recorded in the L3T JET product as `ETdaily`. The standard deviation between these multiple estimates of ET is considered the uncertainty for the SBG evapotranspiration product, as `ETinstUncertainty`. Note that the ETdaily product represents the integrated ET between sunrise and sunset.

#### 2.6.6. AquaSEBS Water Surface Evaporation

For water surface pixels identified using the NASADEM Surface Water Body extent, the ECOSTRESS Collection 3 processing chain implements the AquaSEBS (Aquatic Surface Energy Balance System) model developed by Abdelrady et al. (2016) and validated by Fisher et al. (2023). Water surface evaporation is calculated using a physics-based approach that combines the equilibrium temperature model for water heat flux with the Priestley-Taylor equation for evaporation estimation.

**References**: 
- Abdelrady, A., Timmermans, J., Vekerdy, Z., Salama, M.S. (2016). Surface Energy Balance of Fresh and Saline Waters: AquaSEBS. *Remote Sensing*, 8, 583. https://doi.org/10.3390/rs8070583
- Fisher, J.B., Dohlen, M.B., Halverson, G.H., Collison, J.W., Hook, S.J., Hulley, G.C. (2023). Remotely sensed terrestrial open water evaporation. *Scientific Reports*, 13, 8217. https://doi.org/10.1038/s41598-023-34921-2

**Repository**: [AquaSEBS](https://github.com/JPL-Evapotranspiration-Algorithms/AquaSEBS)

#### Methodology

The AquaSEBS model implements the surface energy balance equation specifically adapted for water bodies:

$$R_n = LE + H + W$$

Where the water heat flux (W) is calculated using the equilibrium temperature model:

$$W = \beta \times (T_e - WST)$$

The key parameters include:
- **Temperature difference**: $T_n = 0.5 \times (WST - T_d)$ where WST is water surface temperature and $T_d$ is dew point temperature
- **Evaporation efficiency**: $\eta = 0.35 + 0.015 \times WST + 0.0012 \times T_n^2$
- **Thermal exchange coefficient**: $\beta = 4.5 + 0.05 \times WST + (\eta + 0.47) \times S$
- **Equilibrium temperature**: $T_e = T_d + \frac{SW_{net}}{\beta}$

Latent heat flux is then calculated using the Priestley-Taylor equation with α = 1.26 for water surfaces:

$$LE = \alpha \times \frac{\Delta}{\Delta + \gamma} \times (R_n - W)$$

#### Validation and Accuracy

The AquaSEBS methodology has been extensively validated against 19 in situ open water evaporation sites worldwide spanning multiple climate zones. Performance metrics include:

**Daily evaporation estimates:**
- **MODIS-based**: r² = 0.47, RMSE = 1.5 mm/day (41% of mean), Bias = 0.19 mm/day
- **Landsat-based**: r² = 0.56, RMSE = 1.2 mm/day (38% of mean), Bias = -0.8 mm/day

**Instantaneous estimates (controlled for high wind events >7.5 m/s):**
- **Correlation**: r² = 0.71
- **RMSE**: 53.7 W/m² (38% of mean)
- **Bias**: -19.1 W/m² (13% of mean)

The model demonstrates particular strength in water-limited environments and performs well across spatial scales from 30m (Landsat) to 1km (MODIS) resolution.

Water surface evaporation estimates are included in the `ETdaily` layer in mm per day, integrated over the daylight period from sunrise to sunset.

### 2.7. L4T ESI and WUE Products

The PT-JPL-SM model generates estimates of both actual and potential instantaneous ET. The potential evapotranspiration (PET) estimate represents the maximum expected ET if there were no water stress to plants on the ground. The ratio of the actual ET estimate to the PET estimate forms an index representing the water stress of plants, with zero being fully stressed with no observable ET and one being non-stressed with ET reaching PET. These ESI and PET estimates are distributed in the L4T ESI product as listed in Table 4.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ESI | Evaporative Stress Index | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| PET | Potential Evapotranspiration | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

*Table 4. Listing of the L4T ESI data layers.*

The BESS-JPL GPP estimate represents the amount of carbon that plants are taking in. The transpiration component of PT-JPL-SM represents the amount of water that plants are releasing. The BESS-JPL GPP is divided by the PT-JPL-SM transpiration to estimate water use efficiency (WUE), the ratio of grams of carbon that plants take in to kilograms of water that plants release. These WUE and GPP estimates are distributed in the L4T WUE product as listed in Table 5.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| WUE | Water Use Efficiency | float32 | $$\text{g C kg}^{-1} \text{H}_2\text{O}$$ | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| GPP | Gross Primary Production | float32 | $$\mu\text{mol m}^{-2} \text{s}^{-1}$$ | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | uint8 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

*Table 5. Listing of the L4T WUE data layers.*

## 3. Theory

The JPL evapotranspiration (JET) data ensemble provides a robust estimation of ET from multiple ET models. The ET ensemble incorporates ET data from four algorithms: Priestley Taylor-Jet Propulsion Laboratory model with soil moisture (PT-JPL-SM), the Penman Monteith-Jet Propulsion Laboratory model (PM-JPL), Surface Temperature Initiated Closure-Jet Propulsion Laboratory model (STIC-JPL), and the Breathing Earth System Simulator-Jet Propulsion Laboratory model (BESS-JPL). We present descriptions of these models here, inherited from the ECOSTRESS mission, as candidates for ECOSTRESS L3 evapotranspiration processing.

## 4. Cal/Val

The ECOSTRESS Collection 3 evapotranspiration products have been validated against flux tower measurements from the FLUXNET network as documented in Pierrat et al. (2025). The validation study evaluated the performance of the L3T JET ensemble evapotranspiration product across multiple biomes and climate conditions.

### 4.1. Validation Methodology

The validation approach compared ECOSTRESS ET estimates with ground-based eddy covariance flux tower measurements from FLUXNET sites. The study assessed:

- **Temporal accuracy**: Agreement between satellite-derived ET and tower measurements at instantaneous and daily timescales
- **Spatial representativeness**: Performance across different land cover types and climate zones
- **Seasonal patterns**: Ability to capture seasonal variations in evapotranspiration
- **Model ensemble performance**: Comparison of individual model components versus the ensemble median

### 4.2. Key Validation Results

The validation demonstrated that the ECOSTRESS Collection 3 ET ensemble:

- Shows strong correlation with flux tower measurements (R² > 0.7) across most biomes
- Captures the diurnal and seasonal patterns of evapotranspiration effectively
- Performs well in water-limited ecosystems where thermal stress indicators are most valuable
- Benefits from the ensemble approach, with the median estimate generally outperforming individual models
- Maintains accuracy across the range of spatial scales from 70m pixels to flux tower footprints

### 4.3. Performance by Biome

The validation results indicate varying performance across different ecosystem types:

- **Croplands**: Excellent agreement during growing season, capturing irrigation and phenological patterns
- **Forests**: Good performance in temperate and boreal forests, with some challenges in dense tropical canopies
- **Grasslands**: Strong performance in both natural and managed grassland systems
- **Shrublands**: Reliable estimates in semi-arid regions where thermal stress is prevalent

The study confirms that ECOSTRESS Collection 3 ET products provide reliable estimates suitable for water resource management, agricultural monitoring, and ecosystem research applications.

#### Acknowledgements 

We would like to thank Joshua Fisher as the initial science lead of the ECOSTRESS mission and PI of the ROSES project to re-design the ECOSTRESS products.

We would like to thank Adam Purdy for contributing the PT-JPL-SM model.

We would like to thank Kaniska Mallick for contributing the STIC model.

## References

- Allen, R.G., Tasumi, M., & Trezza, R. (2007). "Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC)—Model." *Journal of Irrigation and Drainage Engineering*, 133(4), 380-394. https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)
- Brutsaert, W. (1982). *Evaporation into the Atmosphere: Theory, History, and Applications*. Springer Netherlands. https://doi.org/10.1007/978-94-017-1497-6
- Fisher, J.B., Tu, K.P., Baldocchi, D.D. (2008). Global estimates of the land–atmosphere water flux based on monthly AVHRR and ISLSCP-II data, validated at 16 FLUXNET sites. *Remote Sensing of Environment*, 112(3), 901-919. https://doi.org/10.1016/j.rse.2007.06.025
- Monteith, J.L. (1965). "Evaporation and Environment." *Symposia of the Society for Experimental Biology*, 19, 205-234.
- Penman, H.L. (1948). "Natural Evaporation from Open Water, Bare Soil and Grass." *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 193(1032), 120-145. https://doi.org/10.1098/rspa.1948.0037
- Pierrat, Z., et al. (2025). "Validation of ECOSTRESS Collection 3 Evapotranspiration Products Using FLUXNET Measurements." *Remote Sensing of Environment* (in press).
