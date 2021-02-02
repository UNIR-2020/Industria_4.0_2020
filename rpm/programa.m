inputs_rpm=inputs_rpm';
tar_rpm=tar_rpm';
y=tar_rpm;
y_est=net_rpm(inputs_rpm');
plot(y,'b','linewidth',4);
hold on;
grid on;
plot(y_est,'r','linewidth',2)
legend('ORIGINAL','RED NEURONAL')
