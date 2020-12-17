library(unmarked)

data <- read.csv(system.file("csv", "widewt.csv", package = "unmarked"))

y <- data[ ,2:4]

siteCovs <-  data[ ,5:7]

obsCovs <- list(date = data[ ,8:10],
                ivel = data[ ,11:13])

umf <- unmarkedFrameOccu(y = y, siteCovs = siteCovs, obsCovs = obsCovs)

fm1 <- occu(formula = ~ date + ivel
                      ~ forest + elev + length,
            data = umf)

write.csv(coef(fm1, 'state'), 'state_coefs_unmarked.csv')
write.csv(coef(fm1, 'det'), 'det_coefs_unmarked.csv')
write.csv(data, 'data.csv')
