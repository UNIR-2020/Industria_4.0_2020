inputs_vibracion=inputs_vibracion';
tar_vibracion=tar_vibracion';
y=tar_vibracion;
y_est=net_vibracion(inputs_vibracion');
plot(y,'b','linewidth',4);
hold on;
grid on;
plot(y_est,'r','linewidth',2)
legend('ORIGINAL','RED NEURONAL')
