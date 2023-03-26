clear
data = load('p1n00');
marker_1_x = data(:,1);
order = 30;
x = 1:length(marker_1_x);

p = polyfit(x, marker_1_x, order);

fit = zeros(1,length(marker_1_x));
for i = 0:order
    fit = fit + p(i+1)*x.^(order-i);
end

figure(1)
plot(x,marker_1_x,'.')
hold on
plot(x, fit,'-')

data_zero_mean = marker_1_x - fit';
var = var(data_zero_mean); % use this for R
mu = mean(data_zero_mean);

figure(2)
plot(data_zero_mean)