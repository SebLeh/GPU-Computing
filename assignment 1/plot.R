diagrams <- c("64/16384", "128/8192", "256/4096", "512/2048", "1024/1024")
diagrams2 <- c("64/16384", "128/8192", "256/4096", "512/2048", "1024/1024")

GTX1060debug <- c(0.0863739, 0.0855903, 0.0854853, 0.0874706, 0.0856234)
GTX1060release <- c(0.0862561, 0.0855737, 0.0855795, 0.0854462, 0.08566469)
GT540Mdebug <- c(0.956592, 0.588792, 0.507572, 0.515847, 0.88393)
GT540Mrelease <- c(0.957588, 0.58691, 0.506153, 0.512728, 0.885047)

data <- data.frame(diagrams, GTX1060debug, GTX1060release)
data2 <- data.frame(diagrams2, GT540Mdebug, GT540Mrelease)
p1 <- plot(data, x = ~diagrams, y = ~GTX1060debug, type = 'bar', name = 'GTX 1060')
