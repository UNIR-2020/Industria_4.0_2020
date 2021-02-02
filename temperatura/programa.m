inputs_temperatura=inputs_temperatura';
tar_temperatura=tar_temperatura';
y=tar_temperatura;
y_est=net_temperatura(inputs_temperatura');
plot(y,'b','linewidth',4);
hold on;
grid on;
plot(y_est,'r','linewidth',2)
legend('ORIGINAL','RED NEURONAL')
