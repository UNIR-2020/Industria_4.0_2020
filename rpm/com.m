normal=load('rpm_total_buenos.txt'); 
falla=load('rpm_total_falla.txt'); 
input=[falla;normal]; %% mezclasmos los datos de falla y los normales para obtener las entradas
targets=[ones(3000,1);zeros(3000,1)]; %se crean los objetivos con la mezcla de la mitad de valores con 1 que significa falla y la otra mitad con 0 que no hay falla
%%mezclamos la datos
save('input','input');
save('targets','targets');

