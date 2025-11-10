from ECOv002_granules import L2TLSTE

filename = "/Users/halverso/data/ECOv003_example/ECOv002_L2T_LSTE_35698_014_11SPS_20241022T183627_0713_01"
print(filename)
granule = L2TLSTE(filename)
print(granule.time_UTC)
