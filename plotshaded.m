function varargout = plotshaded(x,y,cstr_ci,cstr_mu)
% Modified from Jakob Voigts's original script: http://jvoigts.scripts.mit.edu/blog/nice-shaded-plots/
% x: x coordinates
% y: either just one y vector, or 2xN or 3xN matrix of y-data
% fstr: format ('r' or 'b--' etc)
%
% example
% x=[-10:.1:10];plotshaded(x,[sin(x.*1.1)+1;sin(x*.9)-1],'r');
 
if size(y,1)>size(y,2)
    y=y';
end
 
if size(y,1)==1 % just plot one line
    plot(x,y,fstr);
end
 
if size(y,1)==2 %plot shaded area
    px=[x,fliplr(x)]; % make closed patch
    py=[y(1,:), fliplr(y(2,:))];
    patch(px,py,1,'FaceColor',cstr_ci,'EdgeColor','none');
end
 
if size(y,1)==3 % also draw mean
    px=[x,fliplr(x)];
    py=[y(1,:), fliplr(y(3,:))];
    hold on;
    plot(x,y(2,:),cstr_mu,'linewidth',2);
    patch(px,py,1,'FaceColor',cstr_ci,'EdgeColor','none');
end
 
alpha(.2); % make patch transparent